from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from conftest import MLXTrainerFactory
from finetune.mlx_trainer import MLXLoRATrainer


def _patch_job_id_datetime(mocker: MockerFixture) -> MagicMock:
    mock_datetime: MagicMock = mocker.patch("finetune._common_utils.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return mock_datetime


@dataclass
class PrepareDataMocks:
    training_data_handler: MagicMock
    path: MagicMock
    logger: MagicMock


@dataclass
class WriteConfigMocks:
    to_mlx_config: MagicMock
    yaml: MagicMock


@dataclass
class RunMocks:
    prepare_data: MagicMock
    write_config: MagicMock
    train: MagicMock
    print: MagicMock


@dataclass
class PostProcessMocks:
    path: MagicMock
    fuse: MagicMock
    convert: MagicMock
    upload_model_folder: MagicMock
    archive_and_upload: MagicMock


@pytest.fixture
def subprocess_run(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("finetune.mlx_trainer.subprocess.run")


@pytest.fixture
def prepare_data_mocks(mocker: MockerFixture) -> PrepareDataMocks:
    mock_path: MagicMock = mocker.patch("finetune.mlx_trainer.Path")
    # train_jsonl.read_text() -> three non-empty lines
    mock_path.return_value.__truediv__.return_value.read_text.return_value = (
        "a\nb\nc\n"
    )
    return PrepareDataMocks(
        mocker.patch("finetune.mlx_trainer.TrainingDataHandler.prepare"),
        mock_path,
        mocker.patch("finetune.mlx_trainer.logger"),
    )


@pytest.fixture
def write_config_mocks(mocker: MockerFixture) -> WriteConfigMocks:
    # Real filesystem writes happen under tmp_path (the test passes work_dir=tmp_path),
    # so only the translator and the YAML dump are mocked.
    return WriteConfigMocks(
        mocker.patch(
            "finetune.mlx_trainer.to_mlx_config",
            return_value={"model": "baz"},
        ),
        mocker.patch("finetune.mlx_trainer.yaml.safe_dump"),
    )


@pytest.fixture
def run_mocks(mocker: MockerFixture) -> RunMocks:
    return RunMocks(
        mocker.patch.object(
            MLXLoRATrainer, "prepare_data", return_value="data/models/foo/data"
        ),
        mocker.patch.object(
            MLXLoRATrainer, "_write_config", return_value="resolved.yaml"
        ),
        mocker.patch.object(MLXLoRATrainer, "train"),
        mocker.patch("builtins.print"),
    )


@pytest.fixture
def post_process_mocks(mocker: MockerFixture) -> PostProcessMocks:
    return PostProcessMocks(
        mocker.patch("finetune.mlx_trainer.Path"),
        mocker.patch.object(MLXLoRATrainer, "fuse"),
        mocker.patch.object(MLXLoRATrainer, "convert"),
        mocker.patch("finetune.mlx_trainer.upload_model_folder"),
        mocker.patch("finetune.mlx_trainer.archive_and_upload"),
    )


@pytest.fixture
def model_card() -> MagicMock:
    mock: MagicMock = MagicMock()
    mock.model_name = "foo"
    mock.major = 1
    mock.minor = 2
    mock.patch = 3
    return mock


class TestConstructor:
    # __init__ side effects: job_id, the derived local work paths, base_model from config, and
    # num_samples starting unset (it is filled in by prepare_data). datetime mocked for job_id.
    def test_init_sets_job_id(
        self, mocker: MockerFixture, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        _patch_job_id_datetime(mocker)
        assert make_mlx_trainer(description="grault").job_id == "20260101-120000-grault"

    def test_init_sets_local_paths(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer(
            description="bar", work_dir="work"
        )
        assert trainer.model_folder == "work/bar"
        assert trainer.data_dir == "work/bar/data"
        assert trainer.adapter_dir == "work/bar/adapter"
        assert trainer.target_dir == "work/bar/target"
        assert trainer.mlx_dir == "work/bar/mlx"
        assert (
            trainer.resolved_config_path == "work/bar/mlx_lora_config.resolved.yaml"
        )

    def test_init_loads_base_model_from_config(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        assert make_mlx_trainer().base_model == "baz"

    def test_init_num_samples_unset(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        assert make_mlx_trainer().num_samples is None


class TestPrepareData:
    # Delegates to TrainingDataHandler.prepare, records num_samples from the line count, and
    # returns the local data dir. TrainingDataHandler/Path/line-count mocked.
    def test_calls_training_data_handler_prepare(
        self, prepare_data_mocks: PrepareDataMocks, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer(
            training_batch_names=["batch-a"],
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "x"},
            description="foo",
            work_dir="work",
        )
        trainer.prepare_data()
        prepare_data_mocks.training_data_handler.assert_called_once_with(
            schema=trainer.schema,
            system_prompt="foo",
            training_batch_names=["batch-a"],
            bucket="foo-bar",
            s3_prefix="trainingdata",
            output_file=str(
                prepare_data_mocks.path.return_value.__truediv__.return_value
            ),
            region="foo-bar-1",
            shuffle=True,
        )

    def test_sets_num_samples_from_line_count(
        self, prepare_data_mocks: PrepareDataMocks, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer()
        trainer.prepare_data()
        assert trainer.num_samples == 3

    def test_returns_data_dir(
        self, prepare_data_mocks: PrepareDataMocks, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer(description="foo", work_dir="work")
        assert trainer.prepare_data() == "work/foo/data"


class TestWriteConfig:
    # _write_config requires num_samples (else raises), injects the data/adapter paths into the
    # resolved config, and returns its path. to_mlx_config/yaml mocked; uses tmp_path for writes.
    def test_raises_when_num_samples_unset(
        self, write_config_mocks: WriteConfigMocks, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer()
        with pytest.raises(ValueError, match="prepare_data must run before"):
            trainer._write_config("data/dir")

    def test_injects_data_and_adapter_path(
        self,
        tmp_path: Path,
        write_config_mocks: WriteConfigMocks,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer(
            description="foo", work_dir=str(tmp_path)
        )
        trainer.num_samples = 5
        trainer._write_config(f"{tmp_path}/foo/data")
        write_config_mocks.to_mlx_config.assert_called_once_with(
            trainer.config, num_samples=5
        )
        dumped = write_config_mocks.yaml.call_args[0][0]
        assert dumped["data"] == f"{tmp_path}/foo/data"
        assert dumped["adapter_path"] == f"{tmp_path}/foo/adapter"

    def test_returns_resolved_config_path(
        self,
        tmp_path: Path,
        write_config_mocks: WriteConfigMocks,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer(
            description="foo", work_dir=str(tmp_path)
        )
        trainer.num_samples = 5
        assert (
            trainer._write_config(f"{tmp_path}/foo/data")
            == f"{tmp_path}/foo/mlx_lora_config.resolved.yaml"
        )


class TestTrain:
    # train() shells out to `mlx_lm.lora --config <path>`. subprocess.run mocked.
    def test_calls_subprocess_run(
        self, subprocess_run: MagicMock, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        make_mlx_trainer().train("resolved.yaml")
        subprocess_run.assert_called_once_with(
            ["mlx_lm.lora", "--config", "resolved.yaml"], check=True
        )


class TestRun:
    # Orchestration: run() chains prepare_data -> _write_config -> train and returns the job_id.
    # The three steps mocked; datetime mocked for the job_id assertion.
    def test_orchestrates_prepare_write_train(
        self, run_mocks: RunMocks, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        make_mlx_trainer().run()
        run_mocks.prepare_data.assert_called_once()
        run_mocks.write_config.assert_called_once_with("data/models/foo/data")
        run_mocks.train.assert_called_once_with("resolved.yaml")

    def test_returns_job_id(
        self, mocker: MockerFixture, run_mocks: RunMocks, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        _patch_job_id_datetime(mocker)
        trainer: MLXLoRATrainer = make_mlx_trainer(description="foo")
        assert trainer.run() == "20260101-120000-foo"


class TestFuse:
    # Two branches: an already-fused target short-circuits without shelling out; otherwise fuse()
    # runs `mlx_lm.fuse` with the right args. Path.exists and subprocess.run mocked.
    def test_existing_model_returns_true_without_subprocess(
        self, subprocess_run: MagicMock, mocker: MockerFixture, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        mocker.patch(
            "finetune.mlx_trainer.Path.exists", return_value=True
        )
        assert make_mlx_trainer().fuse("target")
        subprocess_run.assert_not_called()

    def test_calls_mlx_lm_fuse(
        self, subprocess_run: MagicMock, mocker: MockerFixture, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        mocker.patch("finetune.mlx_trainer.Path.exists", return_value=False)
        trainer: MLXLoRATrainer = make_mlx_trainer(description="foo", work_dir="work")
        trainer.fuse("target")
        subprocess_run.assert_called_once_with(
            [
                "mlx_lm.fuse",
                "--model",
                "baz",
                "--adapter-path",
                "work/foo/adapter",
                "--save-path",
                "target",
            ],
            check=True,
        )


class TestConvert:
    # quantize branch: convert() omits the -q flag when quantize is None, appends it otherwise.
    # subprocess.run mocked.
    def test_without_quantize(
        self, subprocess_run: MagicMock, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        make_mlx_trainer(quantize=None).convert("target", "mlx")
        subprocess_run.assert_called_once_with(
            ["mlx_lm.convert", "--hf-path", "target", "--mlx-path", "mlx"], check=True
        )

    def test_with_quantize_appends_flag(
        self, subprocess_run: MagicMock, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        make_mlx_trainer(quantize="q4").convert("target", "mlx")
        subprocess_run.assert_called_once_with(
            ["mlx_lm.convert", "--hf-path", "target", "--mlx-path", "mlx", "-q", "q4"],
            check=True,
        )


class TestPostProcess:
    # Orchestration + branch coverage: fuse->(convert?)->upload ordering, fuse/convert failures
    # raise, the quantize opt-in for convert, and the push_public opt-in. fuse/convert/upload mocked.
    def test_calls_fuse_and_upload(
        self,
        post_process_mocks: PostProcessMocks,
        model_card: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        post_process_mocks.fuse.return_value = True
        make_mlx_trainer(
            aws_config={"bucket": "baz", "region": "qux", "role": "x"}
        ).post_process(model_card)
        post_process_mocks.fuse.assert_called_once()
        post_process_mocks.upload_model_folder.assert_called_once_with(
            target_folder=str(post_process_mocks.path.return_value),
            model_card=model_card,
            region="qux",
            bucket="baz",
        )

    def test_fuse_fails_raises_value_error(
        self,
        post_process_mocks: PostProcessMocks,
        model_card: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        post_process_mocks.fuse.return_value = False
        with pytest.raises(ValueError, match="fusing with base model failed"):
            make_mlx_trainer().post_process(model_card)

    def test_no_quantize_does_not_convert(
        self,
        post_process_mocks: PostProcessMocks,
        model_card: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        post_process_mocks.fuse.return_value = True
        make_mlx_trainer(quantize=None).post_process(model_card)
        post_process_mocks.convert.assert_not_called()

    def test_quantize_calls_convert(
        self,
        post_process_mocks: PostProcessMocks,
        model_card: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        post_process_mocks.fuse.return_value = True
        post_process_mocks.convert.return_value = True
        make_mlx_trainer(quantize="q4").post_process(model_card)
        post_process_mocks.convert.assert_called_once()

    def test_convert_fails_raises_value_error(
        self,
        post_process_mocks: PostProcessMocks,
        model_card: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        post_process_mocks.fuse.return_value = True
        post_process_mocks.convert.return_value = False
        with pytest.raises(ValueError, match="converting to MLX format failed"):
            make_mlx_trainer(quantize="q4").post_process(model_card)

    def test_default_does_not_push_public(
        self,
        post_process_mocks: PostProcessMocks,
        model_card: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        post_process_mocks.fuse.return_value = True
        make_mlx_trainer().post_process(model_card)
        post_process_mocks.archive_and_upload.assert_not_called()

    def test_push_public_calls_archive_and_upload(
        self,
        post_process_mocks: PostProcessMocks,
        model_card: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        post_process_mocks.fuse.return_value = True
        make_mlx_trainer(
            model_name="grault",
            aws_config={"bucket": "baz", "region": "qux", "role": "x"},
        ).post_process(model_card, push_public=True)
        post_process_mocks.archive_and_upload.assert_called_once_with(
            target_folder=str(post_process_mocks.path.return_value),
            model_card=model_card,
            model_name="grault",
            region="qux",
        )
