from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from conftest import HuggingFaceLoRATrainerFixture, TrainerFactory
from finetune.hf_trainer import HuggingFaceLoRATrainer
from finetune.trainer import LoRATrainer

# Expected hyperparameters for the shared config fixture (see conftest.CONFIG_YAML),
# i.e. FinetuneConfig.load(config_path).to_hf_hyperparameters().
EXPECTED_HYPERPARAMETERS = {
    "base_model": "baz",
    "num_epochs": 2,
    "learning_rate": 0.0002,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj",
    "per_device_train_batch_size": 4,
    "max_seq_length": 2048,
}


@dataclass
class ConstructorMocks:
    datetime: MagicMock


@dataclass
class PrepareDataMocks:
    training_data_handler: MagicMock
    aws: MagicMock
    datetime: MagicMock


@dataclass
class LaunchJobMocks:
    huggingface: MagicMock
    path: MagicMock
    datetime: MagicMock


@dataclass
class RunMocks:
    prepare_data: MagicMock
    launch_job: MagicMock
    datetime: MagicMock


@dataclass
class DownloadOutputMocks:
    path: MagicMock
    aws: MagicMock
    tarfile: MagicMock


@dataclass
class PostProcessMocks:
    path: MagicMock
    download_output: MagicMock
    merge: MagicMock
    upload_model_folder: MagicMock
    archive_and_upload: MagicMock
    datetime: MagicMock


def _patch_job_id_datetime(mocker: MockerFixture) -> MagicMock:
    """Make _make_job_id deterministic: it lives on the base trainer now."""
    mock_datetime: MagicMock = mocker.patch("finetune.trainer.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return mock_datetime


@pytest.fixture
def constructor_mocks(mocker: MockerFixture) -> ConstructorMocks:
    return ConstructorMocks(_patch_job_id_datetime(mocker))


@pytest.fixture
def prepare_data_mocks(mocker: MockerFixture) -> PrepareDataMocks:
    mock_datetime: MagicMock = _patch_job_id_datetime(mocker)
    return PrepareDataMocks(
        mocker.patch(
            "finetune.trainer.TrainingDataHandler.prepare",
            return_value="train.jsonl",
        ),
        mocker.patch("finetune.hf_estimator.AWS.upload_file"),
        mock_datetime,
    )


@pytest.fixture
def launch_job_mocks(mocker: MockerFixture) -> LaunchJobMocks:
    mock_datetime: MagicMock = _patch_job_id_datetime(mocker)
    mock_hf: MagicMock = mocker.patch("finetune.hf_trainer.HuggingFace")
    mock_hf.return_value.latest_training_job.name = "mesa-foo-bar"
    mock_path: MagicMock = mocker.patch("finetune.hf_trainer.Path")
    mock_path.return_value.parent.__truediv__.return_value = "/foo/scripts"
    return LaunchJobMocks(
        mock_hf,
        mock_path,
        mock_datetime,
    )


@pytest.fixture
def run_mocks(mocker: MockerFixture) -> RunMocks:
    mock_datetime: MagicMock = _patch_job_id_datetime(mocker)
    return RunMocks(
        mocker.patch.object(
            HuggingFaceLoRATrainer, "prepare_data", return_value="s3://foo/bar"
        ),
        mocker.patch.object(
            HuggingFaceLoRATrainer, "launch_job", return_value="mesa-foo-bar"
        ),
        mock_datetime,
    )


@pytest.fixture
def download_output_mocks(mocker: MockerFixture) -> DownloadOutputMocks:
    return DownloadOutputMocks(
        mocker.patch("finetune.hf_trainer.Path"),
        mocker.patch("finetune.hf_trainer.AWS.download_file"),
        mocker.patch("finetune.hf_trainer.tarfile.open"),
    )


@pytest.fixture
def post_process_mocks(mocker: MockerFixture) -> PostProcessMocks:
    mock_datetime: MagicMock = _patch_job_id_datetime(mocker)
    return PostProcessMocks(
        mocker.patch("finetune.hf_trainer.Path"),
        mocker.patch.object(HuggingFaceLoRATrainer, "download_output"),
        mocker.patch.object(HuggingFaceLoRATrainer, "merge"),
        mocker.patch.object(LoRATrainer, "_upload_model_folder"),
        mocker.patch.object(LoRATrainer, "_archive_and_upload"),
        mock_datetime,
    )


class TestConstructor:
    # __init__ side effects: job_id from timestamp+description, the three derived S3 paths,
    # base_model loaded from config, and config->HF hyperparameter translation. datetime mocked.
    def test_init_sets_job_id_with_timestamp_and_description(
        self, constructor_mocks: ConstructorMocks, make_trainer: TrainerFactory
    ) -> None:
        assert (
            make_trainer(description="grault").get_job_id() == "20260101-120000-grault"
        )

    def test_init_sets_s3_paths(
        self, constructor_mocks: ConstructorMocks, make_trainer: TrainerFactory
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = make_trainer(
            aws_config={"bucket": "xyzzy", "region": "thud", "role": "wibble"},
            description="wobble",
        )
        assert trainer.get_s3_input_path() == "jobs/train/20260101-120000-wobble/input"
        assert (
            trainer.get_s3_output_path() == "jobs/train/20260101-120000-wobble/output"
        )
        assert (
            trainer.get_s3_full_output_path()
            == "s3://xyzzy/jobs/train/20260101-120000-wobble/output"
        )

    def test_init_loads_base_model_from_config(
        self, constructor_mocks: ConstructorMocks, make_trainer: TrainerFactory
    ) -> None:
        assert make_trainer().base_model == "baz"

    def test_init_translates_config_to_hyperparameters(
        self, constructor_mocks: ConstructorMocks, make_trainer: TrainerFactory
    ) -> None:
        assert make_trainer().hyperparameters == EXPECTED_HYPERPARAMETERS


class TestPrepareData:
    # Stages training data to S3: delegates to TrainingDataHandler.prepare, uploads via AWS with
    # the right kwargs, and returns the s3:// input path. TrainingDataHandler/AWS/datetime mocked.
    def test_prepare_data_calls_training_data_handler_prepare(
        self, prepare_data_mocks: PrepareDataMocks, make_trainer: TrainerFactory
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = make_trainer(
            training_batch_names=["20260101-120000_corge-quux"],
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "foo"},
        )
        trainer.prepare_data()
        prepare_data_mocks.training_data_handler.assert_called_once_with(
            schema=trainer.schema,
            system_prompt="foo",
            training_batch_names=["20260101-120000_corge-quux"],
            bucket="foo-bar",
            s3_prefix="trainingdata",
            output_file="train.jsonl",
            region="foo-bar-1",
            shuffle=True,
        )

    def test_prepare_data_calls_aws_upload_file(
        self, prepare_data_mocks: PrepareDataMocks, make_trainer: TrainerFactory
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = make_trainer(
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "bar"},
        )
        trainer.prepare_data()
        prepare_data_mocks.aws.assert_called_once_with(
            region_name="foo-bar-1",
            file_name="train.jsonl",
            bucket="foo-bar",
            object_name="train.jsonl",
            path="jobs/train/20260101-120000-foo/input",
        )

    def test_prepare_data_returns_s3_path(
        self, prepare_data_mocks: PrepareDataMocks, make_trainer: TrainerFactory
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = make_trainer(
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "baz"},
        )
        assert (
            trainer.prepare_data()
            == "s3://foo-bar/jobs/train/20260101-120000-foo/input"
        )


class TestLaunchJob:
    # Configures and fires the SageMaker HuggingFace estimator: constructor kwargs, fit() call
    # with the training path, and returns the launched job name. HuggingFace/Path/datetime mocked.
    def test_launch_job_creates_huggingface_estimator(
        self, launch_job_mocks: LaunchJobMocks, make_trainer: TrainerFactory
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = make_trainer(
            aws_config={
                "bucket": "foo-bar",
                "region": "foo-bar-1",
                "role": "arn:aws:iam::123:role/foo",
            },
        )
        trainer.launch_job("s3://foo-bar/input")
        launch_job_mocks.huggingface.assert_called_once_with(
            entry_point="train_lora.py",
            source_dir="/foo/scripts",
            code_location="s3://foo-bar/jobs/train/20260101-120000-foo",
            role="arn:aws:iam::123:role/foo",
            instance_type="foo.bar1.2baz",
            instance_count=1,
            transformers_version="1.23",
            pytorch_version="1.2",
            py_version="foo123",
            output_path="s3://foo-bar/jobs/train/20260101-120000-foo/output",
            base_job_name="mesa-20260101-120000-foo",
            hyperparameters=EXPECTED_HYPERPARAMETERS,
        )

    def test_launch_job_calls_fit_with_training_path(
        self, launch_job_mocks: LaunchJobMocks, make_trainer: TrainerFactory
    ) -> None:
        make_trainer().launch_job("s3://bucket/path/to/data")
        launch_job_mocks.huggingface.return_value.fit.assert_called_once_with(
            {"training": "s3://bucket/path/to/data"}, wait=False
        )

    def test_launch_job_returns_job_name(
        self, launch_job_mocks: LaunchJobMocks, make_trainer: TrainerFactory
    ) -> None:
        assert make_trainer().launch_job("s3://bucket/input") == "mesa-foo-bar"


class TestRun:
    # Orchestration: run() calls prepare_data, feeds its result to launch_job, returns the job
    # name, and records it as last_job_name. prepare_data/launch_job/datetime mocked.
    def test_run_calls_prepare_data(
        self, run_mocks: RunMocks, make_trainer: TrainerFactory
    ) -> None:
        make_trainer().run()
        run_mocks.prepare_data.assert_called_once()

    def test_run_calls_launch_job_with_prepare_data_result(
        self, run_mocks: RunMocks, make_trainer: TrainerFactory
    ) -> None:
        make_trainer().run()
        run_mocks.launch_job.assert_called_once_with("s3://foo/bar")

    def test_run_returns_job_name(
        self, run_mocks: RunMocks, make_trainer: TrainerFactory
    ) -> None:
        assert make_trainer().run() == "mesa-foo-bar"

    def test_run_sets_last_job_name(
        self, run_mocks: RunMocks, make_trainer: TrainerFactory
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = make_trainer()
        trainer.run()
        assert trainer.last_job_name == "mesa-foo-bar"


class TestDownloadOutput:
    # Branch coverage on the model.tar.gz download: cached short-circuit, AWS download kwargs,
    # download-failure raises, and the success path extracts to the parent. Path/AWS/tarfile mocked.
    def test_download_output_cached_short_circuits(
        self, download_output_mocks: DownloadOutputMocks, make_trainer: TrainerFactory
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = True
        assert make_trainer().download_output("foo/bar", "baz/qux", "quux")
        download_output_mocks.aws.assert_not_called()
        download_output_mocks.tarfile.assert_not_called()

    def test_download_output_calls_aws_download_file(
        self, download_output_mocks: DownloadOutputMocks, make_trainer: TrainerFactory
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = True
        trainer: HuggingFaceLoRATrainerFixture = make_trainer(
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "corge"}
        )
        trainer.download_output("grault/garply", "waldo/fred", "plugh")
        download_output_mocks.aws.assert_called_once_with(
            region_name="foo-bar-1",
            bucket="foo-bar",
            file_name=str(download_output_mocks.path.return_value),
            object_name="model.tar.gz",
            path="waldo/fred/plugh/output",
        )

    def test_download_output_download_fails(
        self, download_output_mocks: DownloadOutputMocks, make_trainer: TrainerFactory
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = False
        with pytest.raises(ValueError, match="Failed to download training output"):
            make_trainer().download_output("foo/bar", "baz/qux", "quux")
        download_output_mocks.tarfile.assert_not_called()

    def test_download_output_success(
        self, download_output_mocks: DownloadOutputMocks, make_trainer: TrainerFactory
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = True
        assert make_trainer().download_output("foo/bar", "baz/qux", "quux")
        download_output_mocks.tarfile.assert_called_once_with(
            download_output_mocks.path.return_value, "r:*"
        )
        download_output_mocks.tarfile.return_value.__enter__.return_value.extractall.assert_called_once_with(
            download_output_mocks.path.return_value.parent
        )


class TestPostProcess:
    # Orchestration + branch coverage: download->merge->upload ordering, path/job-name resolution
    # (provided vs default/last), error short-circuits, and the push_public opt-in. AWS/SageMaker mocked.
    @pytest.fixture
    def mock_model_card(self) -> MagicMock:
        mock: MagicMock = MagicMock()
        mock.model_name = "foo"
        mock.major = 1
        mock.minor = 2
        mock.patch = 3
        return mock

    def test_post_process_creates_source_and_target_folders(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        make_trainer(description="foo").post_process(mock_model_card, "bar/baz", "qux")
        post_process_mocks.path.return_value.mkdir.assert_any_call(
            parents=True, exist_ok=True
        )
        assert (
            post_process_mocks.path.return_value.mkdir.call_count == 2
        )  # source + target

    def test_post_process_uses_provided_s3_output_path(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        make_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.download_output.assert_called_once()
        assert post_process_mocks.download_output.call_args[0][1] == "foo/bar"

    def test_post_process_uses_default_s3_output_path_when_none(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        trainer: HuggingFaceLoRATrainerFixture = make_trainer(description="foo")
        trainer.post_process(mock_model_card, None, "bar")
        post_process_mocks.download_output.assert_called_once()
        assert (
            post_process_mocks.download_output.call_args[0][1]
            == "jobs/train/20260101-120000-foo/output"
        )

    def test_post_process_uses_provided_job_name(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        make_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.download_output.assert_called_once()
        assert post_process_mocks.download_output.call_args[0][2] == "baz"

    def test_post_process_uses_last_job_name_when_none(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        trainer: HuggingFaceLoRATrainerFixture = make_trainer()
        trainer.last_job_name = "foo-bar"
        trainer.post_process(mock_model_card, "baz/qux", None)
        post_process_mocks.download_output.assert_called_once()
        assert post_process_mocks.download_output.call_args[0][2] == "foo-bar"

    def test_post_process_no_job_name_raises_value_error(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = make_trainer()
        trainer.last_job_name = None
        with pytest.raises(
            ValueError, match="no last job available and no job name specified"
        ):
            trainer.post_process(mock_model_card, "foo/bar", None)

    def test_post_process_download_fails_raises_value_error(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = False
        with pytest.raises(ValueError, match="downloading low-rank weights failed"):
            make_trainer().post_process(mock_model_card, "foo/bar", "baz")

    def test_post_process_download_fails_does_not_call_merge(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = False
        with pytest.raises(ValueError):
            make_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.merge.assert_not_called()

    def test_post_process_merge_fails_raises_value_error(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = False
        with pytest.raises(ValueError, match="merging with base model failed"):
            make_trainer().post_process(mock_model_card, "foo/bar", "baz")

    def test_post_process_merge_fails_does_not_call_upload(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = False
        with pytest.raises(ValueError):
            make_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.upload_model_folder.assert_not_called()

    def test_post_process_calls_download_output(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        make_trainer(description="foo").post_process(mock_model_card, "bar/baz", "qux")
        post_process_mocks.download_output.assert_called_once_with(
            str(post_process_mocks.path.return_value), "bar/baz", "qux"
        )

    def test_post_process_calls_merge(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        make_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.merge.assert_called_once_with(
            str(post_process_mocks.path.return_value),
            str(post_process_mocks.path.return_value),
        )

    def test_post_process_calls_upload_model_folder(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        make_trainer(
            aws_config={"bucket": "baz", "region": "qux", "role": "quux"}
        ).post_process(mock_model_card, "foo/bar", "fred")
        post_process_mocks.upload_model_folder.assert_called_once_with(
            str(post_process_mocks.path.return_value), mock_model_card
        )

    def test_post_process_default_does_not_push_public(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        make_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.archive_and_upload.assert_not_called()

    def test_post_process_push_public_calls_archive_and_upload(
        self,
        post_process_mocks: PostProcessMocks,
        mock_model_card: MagicMock,
        make_trainer: TrainerFactory,
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        make_trainer(
            model_name="grault",
            aws_config={"bucket": "baz", "region": "qux", "role": "quux"},
        ).post_process(mock_model_card, "foo/bar", "fred", push_public=True)
        post_process_mocks.archive_and_upload.assert_called_once_with(
            str(post_process_mocks.path.return_value), mock_model_card
        )
