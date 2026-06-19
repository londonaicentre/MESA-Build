import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from pytest_mock import MockerFixture

from conftest import MLXTrainerFactory
from finetune.config import FinetuneConfig
from finetune.mlx_trainer import MLXLoRATrainer
from finetune.trainer import LoRATrainer


def _patch_job_id_datetime(mocker: MockerFixture) -> MagicMock:
    mock_datetime: MagicMock = mocker.patch("finetune.trainer.datetime")
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


@dataclass
class TrainMocks:
    subprocess_run: MagicMock
    latest_checkpoint: MagicMock
    inject_resume: MagicMock
    logger: MagicMock


@pytest.fixture
def subprocess_run(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("finetune.mlx_trainer.subprocess.run")


@pytest.fixture
def prepare_data_mocks(mocker: MockerFixture) -> PrepareDataMocks:
    mock_path: MagicMock = mocker.patch("finetune.mlx_trainer.Path")
    train_jsonl: MagicMock = mock_path.return_value.__truediv__.return_value
    # train_jsonl.read_text() -> three non-empty lines; absent by default so the
    # idempotency guard falls through to a real S3 prepare
    train_jsonl.read_text.return_value = "a\nb\nc\n"
    train_jsonl.exists.return_value = False
    return PrepareDataMocks(
        mocker.patch("finetune.trainer.TrainingDataHandler.prepare"),
        mock_path,
        mocker.patch("finetune.mlx_trainer.logger"),
    )


@pytest.fixture
def write_config_mocks(mocker: MockerFixture) -> WriteConfigMocks:
    # Real filesystem writes happen under tmp_path (the test passes work_dir=tmp_path),
    # so only the translator and the YAML dump are mocked.
    return WriteConfigMocks(
        mocker.patch.object(
            FinetuneConfig, "to_mlx_config", return_value={"model": "baz"}
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
        mocker.patch.object(LoRATrainer, "_upload_model_folder"),
        mocker.patch.object(LoRATrainer, "_archive_and_upload"),
    )


@pytest.fixture
def train_mocks(mocker: MockerFixture) -> TrainMocks:
    return TrainMocks(
        mocker.patch("finetune.mlx_trainer.subprocess.run"),
        mocker.patch.object(MLXLoRATrainer, "_latest_checkpoint"),
        mocker.patch.object(MLXLoRATrainer, "_inject_resume"),
        mocker.patch("finetune.mlx_trainer.logger"),
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
        self, mocker: MockerFixture, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        _patch_job_id_datetime(mocker)
        trainer: MLXLoRATrainer = make_mlx_trainer(
            model_name="foo", description="bar", work_dir="work"
        )
        folder = "work/foo/20260101-120000-bar"
        assert trainer.model_folder == folder
        assert trainer.data_dir == f"{folder}/data"
        assert trainer.adapter_dir == f"{folder}/adapter"
        assert trainer.target_dir == f"{folder}/target"
        assert trainer.mlx_dir == f"{folder}/mlx"
        assert trainer.resolved_config_path == f"{folder}/mlx_lora_config.resolved.yaml"

    def test_restore_runtime_recomputes_paths_with_saved_job_id(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        original: MLXLoRATrainer = make_mlx_trainer(model_name="foo", work_dir="work")
        rebuilt: MLXLoRATrainer = MLXLoRATrainer.from_json(original.to_json())
        assert rebuilt.job_id == original.job_id
        assert rebuilt.model_folder == f"work/foo/{original.job_id}"

    def test_init_loads_base_model_from_config(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        assert make_mlx_trainer().base_model == "baz"

    def test_init_num_samples_unset(self, make_mlx_trainer: MLXTrainerFactory) -> None:
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
            base_model="baz",
            max_seq_length=2048,
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
        assert trainer.prepare_data() == trainer.data_dir

    def test_existing_data_skips_prepare(
        self, prepare_data_mocks: PrepareDataMocks, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        prepare_data_mocks.path.return_value.__truediv__.return_value.exists.return_value = True
        make_mlx_trainer().prepare_data()
        prepare_data_mocks.training_data_handler.assert_not_called()


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
        write_config_mocks.to_mlx_config.assert_called_once_with(5)
        dumped = write_config_mocks.yaml.call_args[0][0]
        assert dumped["data"] == f"{tmp_path}/foo/data"
        assert dumped["adapter_path"] == trainer.adapter_dir

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
            == trainer.resolved_config_path
        )


def _write_resolved_config(
    tmp_path: Path, iters: int = 1000, lr_schedule: dict[str, str] | None = None
) -> str:
    config: dict[str, object] = {"iters": iters}
    if lr_schedule is not None:
        config["lr_schedule"] = lr_schedule
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


class TestTrain:
    # train() shells out to `mlx_lm.lora --config <path>`, retrying only a SIGABRT (GPU victim) from the latest checkpoint. subprocess.run / checkpoint / inject mocked.
    def test_success_calls_subprocess_once(
        self,
        tmp_path: Path,
        train_mocks: TrainMocks,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        config = _write_resolved_config(tmp_path)
        train_mocks.subprocess_run.return_value = MagicMock(returncode=0)
        make_mlx_trainer().train(config)
        train_mocks.subprocess_run.assert_called_once_with(
            ["mlx_lm.lora", "--config", config]
        )
        train_mocks.inject_resume.assert_not_called()

    @staticmethod
    def _returncodes(*codes: int) -> list[MagicMock]:
        return [MagicMock(returncode=code) for code in codes]

    def test_sigabrt_retries_from_checkpoint(
        self,
        tmp_path: Path,
        train_mocks: TrainMocks,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        config = _write_resolved_config(tmp_path, 1000)
        train_mocks.latest_checkpoint.return_value = (
            Path("0000100_adapters.safetensors"),
            100,
        )
        train_mocks.subprocess_run.side_effect = TestTrain._returncodes(
            -signal.SIGABRT, 0
        )
        make_mlx_trainer().train(config)
        # remaining iters = original (1000) - completed (100)
        train_mocks.inject_resume.assert_called_once_with(
            config, Path("0000100_adapters.safetensors"), 900
        )
        assert train_mocks.subprocess_run.call_count == 2

    def test_non_sigabrt_does_not_retry(
        self,
        tmp_path: Path,
        train_mocks: TrainMocks,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        train_mocks.latest_checkpoint.return_value = (
            Path("0000100_adapters.safetensors"),
            100,
        )
        train_mocks.subprocess_run.return_value = MagicMock(returncode=1)
        with pytest.raises(subprocess.CalledProcessError):
            make_mlx_trainer().train(_write_resolved_config(tmp_path))
        train_mocks.inject_resume.assert_not_called()

    def test_sigabrt_without_checkpoint_raises(
        self,
        tmp_path: Path,
        train_mocks: TrainMocks,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        train_mocks.latest_checkpoint.return_value = None
        train_mocks.subprocess_run.return_value = MagicMock(returncode=-signal.SIGABRT)
        with pytest.raises(subprocess.CalledProcessError):
            make_mlx_trainer().train(_write_resolved_config(tmp_path))

    def test_exhausts_retries_and_raises(
        self,
        tmp_path: Path,
        train_mocks: TrainMocks,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        train_mocks.latest_checkpoint.return_value = (
            Path("0000100_adapters.safetensors"),
            100,
        )
        train_mocks.subprocess_run.return_value = MagicMock(returncode=-signal.SIGABRT)
        with pytest.raises(subprocess.CalledProcessError):
            make_mlx_trainer().train(_write_resolved_config(tmp_path), max_retries=2)
        assert train_mocks.subprocess_run.call_count == 2


class TestLatestCheckpoint:
    @pytest.fixture
    def trainer(
        self, tmp_path: Path, make_mlx_trainer: MLXTrainerFactory
    ) -> MLXLoRATrainer:
        trainer: MLXLoRATrainer = make_mlx_trainer(work_dir=str(tmp_path))
        Path(trainer.adapter_dir).mkdir(parents=True)
        return trainer

    def test_none_when_no_checkpoints(self, trainer: MLXLoRATrainer) -> None:
        assert trainer._latest_checkpoint() is None

    def test_picks_newest_by_mtime(self, trainer: MLXLoRATrainer) -> None:
        older = Path(trainer.adapter_dir) / "0012800_adapters.safetensors"
        newer = Path(trainer.adapter_dir) / "0000100_adapters.safetensors"
        older.write_text("x")
        os.utime(older, (1000, 1000))
        newer.write_text("x")
        os.utime(newer, (2000, 2000))
        assert trainer._latest_checkpoint() == (newer, 100)

    def test_ignores_unnumbered_final_adapter(self, trainer: MLXLoRATrainer) -> None:
        (Path(trainer.adapter_dir) / "adapters.safetensors").write_text("x")
        assert trainer._latest_checkpoint() is None


class TestInjectResume:
    @pytest.fixture
    def version_mock(self, mocker: MockerFixture) -> MagicMock:
        return mocker.patch(
            "finetune.mlx_trainer.version",
            return_value=MLXLoRATrainer.MLX_LM_VALIDATED_VERSION,
        )

    def test_writes_resume_file_and_iters(
        self,
        tmp_path: Path,
        version_mock: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        config = _write_resolved_config(tmp_path, 1000)
        make_mlx_trainer()._inject_resume(
            config, Path("0000100_adapters.safetensors"), 900
        )
        written = yaml.safe_load(Path(config).read_text())
        assert written["resume_adapter_file"] == "0000100_adapters.safetensors"
        assert written["iters"] == 900

    def test_version_mismatch_raises(
        self,
        tmp_path: Path,
        version_mock: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        version_mock.return_value = "0.99.0"
        with pytest.raises(RuntimeError, match="resume validated for mlx-lm"):
            make_mlx_trainer()._inject_resume(
                _write_resolved_config(tmp_path),
                Path("0000100_adapters.safetensors"),
                900,
            )

    def test_lr_schedule_raises(
        self,
        tmp_path: Path,
        version_mock: MagicMock,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        config = _write_resolved_config(tmp_path, 1000, {"name": "cosine"})
        with pytest.raises(ValueError, match="non-constant lr_schedule"):
            make_mlx_trainer()._inject_resume(
                config, Path("0000100_adapters.safetensors"), 900
            )


class TestResumeTrain:
    def test_injects_then_trains(
        self, tmp_path: Path, mocker: MockerFixture, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        config = _write_resolved_config(tmp_path, 15000)
        mocker.patch.object(
            MLXLoRATrainer,
            "_latest_checkpoint",
            return_value=(Path("0012800_adapters.safetensors"), 12800),
        )
        inject: MagicMock = mocker.patch.object(MLXLoRATrainer, "_inject_resume")
        train: MagicMock = mocker.patch.object(MLXLoRATrainer, "train")
        make_mlx_trainer().resume_train(config)
        # remaining iters = original (15000) - completed (12800)
        inject.assert_called_once_with(
            config, Path("0012800_adapters.safetensors"), 2200
        )
        train.assert_called_once_with(config)

    def test_no_checkpoint_raises(
        self, mocker: MockerFixture, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        mocker.patch.object(MLXLoRATrainer, "_latest_checkpoint", return_value=None)
        with pytest.raises(ValueError, match="no checkpoint to resume from"):
            make_mlx_trainer().resume_train("c.yaml")


class TestCleanup:
    def test_removes_model_folder(
        self, mocker: MockerFixture, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        rmtree: MagicMock = mocker.patch("finetune.mlx_trainer.shutil.rmtree")
        trainer: MLXLoRATrainer = make_mlx_trainer()
        trainer.cleanup()
        rmtree.assert_called_once_with(trainer.model_folder, ignore_errors=True)


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
        self,
        mocker: MockerFixture,
        run_mocks: RunMocks,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        _patch_job_id_datetime(mocker)
        trainer: MLXLoRATrainer = make_mlx_trainer(description="foo")
        assert trainer.run() == "20260101-120000-foo"


class TestFuse:
    # Two branches: an already-fused target short-circuits without shelling out; otherwise fuse()
    # runs `mlx_lm.fuse` with the right args. Path.exists and subprocess.run mocked.
    def test_existing_model_returns_true_without_subprocess(
        self,
        subprocess_run: MagicMock,
        mocker: MockerFixture,
        make_mlx_trainer: MLXTrainerFactory,
    ) -> None:
        mocker.patch("finetune.mlx_trainer.Path.exists", return_value=True)
        assert make_mlx_trainer().fuse("target")
        subprocess_run.assert_not_called()

    def test_calls_mlx_lm_fuse(
        self,
        subprocess_run: MagicMock,
        mocker: MockerFixture,
        make_mlx_trainer: MLXTrainerFactory,
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
                trainer.adapter_dir,
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


class TestSerialise:
    def test_to_dict_extends_with_mlx_fields(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer(work_dir="work", quantize="q4")
        trainer.num_samples = 7
        data = trainer.to_dict()
        assert data["work_dir"] == "work"
        assert data["quantize"] == "q4"
        assert data["num_samples"] == 7

    def test_from_json_round_trips_mlx_fields(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        rebuilt: MLXLoRATrainer = MLXLoRATrainer.from_json(
            make_mlx_trainer(
                description="bar", work_dir="work", quantize="q8"
            ).to_json()
        )
        assert rebuilt.work_dir == "work"
        assert rebuilt.quantize == "q8"
        assert rebuilt.model_folder == f"work/foo/{rebuilt.job_id}"
        assert rebuilt.adapter_dir == f"work/foo/{rebuilt.job_id}/adapter"

    def test_from_json_restores_num_samples(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        trainer: MLXLoRATrainer = make_mlx_trainer()
        trainer.num_samples = 42
        assert MLXLoRATrainer.from_json(trainer.to_json()).num_samples == 42

    def test_from_json_round_trips_none_fields(
        self, make_mlx_trainer: MLXTrainerFactory
    ) -> None:
        rebuilt: MLXLoRATrainer = MLXLoRATrainer.from_json(
            make_mlx_trainer(quantize=None).to_json()
        )
        assert rebuilt.quantize is None
        assert rebuilt.num_samples is None


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
            str(post_process_mocks.path.return_value), model_card
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
            str(post_process_mocks.path.return_value), model_card
        )
