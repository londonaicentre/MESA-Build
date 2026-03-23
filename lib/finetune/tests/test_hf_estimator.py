from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from finetune.hf_estimator import HuggingFaceLoRATrainer
from utils.prompt import BasePromptBuilder
from fixtures import SchemaFixture


class PromptBuilderFixture(BasePromptBuilder):
    def __init__(self) -> None:
        pass

    def build_main_prompt(self) -> str:
        return "foo"


class HuggingFaceLoRATrainerFixture(HuggingFaceLoRATrainer):
    def get_job_id(self) -> str:
        return self.job_id

    def get_s3_input_path(self) -> str:
        return self.s3_input_path

    def get_s3_output_path(self) -> str:
        return self.s3_output_path

    def get_s3_full_output_path(self) -> str:
        return self.s3_full_output_path


@dataclass
class ConstructorMocks:
    datetime: MagicMock


@dataclass
class PrepareDataMocks:
    training_data_handler: MagicMock
    aws: MagicMock
    logger: MagicMock
    datetime: MagicMock


@dataclass
class LaunchJobMocks:
    huggingface: MagicMock
    path: MagicMock
    logger: MagicMock
    datetime: MagicMock


@dataclass
class RunMocks:
    prepare_data: MagicMock
    launch_job: MagicMock
    print: MagicMock
    datetime: MagicMock


@dataclass
class DownloadOutputMocks:
    path: MagicMock
    aws: MagicMock
    tarfile: MagicMock
    datetime: MagicMock


@dataclass
class UploadOutputMocks:
    path: MagicMock
    tarfile: MagicMock
    tarinfo: MagicMock
    io: MagicMock
    aws: MagicMock
    datetime: MagicMock


@dataclass
class PostProcessMocks:
    path: MagicMock
    download_output: MagicMock
    merge: MagicMock
    upload_output: MagicMock
    datetime: MagicMock


@pytest.fixture
def constructor_mocks(mocker: MockerFixture) -> ConstructorMocks:
    mock_datetime: MagicMock = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return ConstructorMocks(mock_datetime)


@pytest.fixture
def prepare_data_mocks(mocker: MockerFixture) -> PrepareDataMocks:
    mock_datetime: MagicMock = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return PrepareDataMocks(
        mocker.patch(
            "finetune.hf_estimator.TrainingDataHandler.prepare",
            return_value="train.jsonl",
        ),
        mocker.patch("finetune.hf_estimator.AWS.upload_file"),
        mocker.patch("finetune.hf_estimator.logger"),
        mock_datetime,
    )


@pytest.fixture
def launch_job_mocks(mocker: MockerFixture) -> LaunchJobMocks:
    mock_datetime: MagicMock = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    mock_hf: MagicMock = mocker.patch("finetune.hf_estimator.HuggingFace")
    mock_hf.return_value.latest_training_job.name = "mesa-foo-bar"
    mock_path: MagicMock = mocker.patch("finetune.hf_estimator.Path")
    mock_path.return_value.parent.__truediv__.return_value = "/foo/scripts"
    return LaunchJobMocks(
        mock_hf,
        mock_path,
        mocker.patch("finetune.hf_estimator.logger"),
        mock_datetime,
    )


@pytest.fixture
def run_mocks(mocker: MockerFixture) -> RunMocks:
    mock_datetime: MagicMock = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return RunMocks(
        mocker.patch.object(
            HuggingFaceLoRATrainer, "prepare_data", return_value="s3://foo/bar"
        ),
        mocker.patch.object(
            HuggingFaceLoRATrainer, "launch_job", return_value="mesa-foo-bar"
        ),
        mocker.patch("builtins.print"),
        mock_datetime,
    )


@pytest.fixture
def download_output_mocks(mocker: MockerFixture) -> DownloadOutputMocks:
    mock_datetime: MagicMock = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return DownloadOutputMocks(
        mocker.patch("finetune.hf_estimator.Path"),
        mocker.patch("finetune.hf_estimator.AWS.download_file"),
        mocker.patch("finetune.hf_estimator.tarfile.open"),
        mock_datetime,
    )


@pytest.fixture
def upload_output_mocks(mocker: MockerFixture) -> UploadOutputMocks:
    mock_datetime: MagicMock = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return UploadOutputMocks(
        mocker.patch("finetune.hf_estimator.Path"),
        mocker.patch("finetune.hf_estimator.tarfile.open"),
        mocker.patch("finetune.hf_estimator.tarfile.TarInfo"),
        mocker.patch("io.BytesIO"),
        mocker.patch("finetune.hf_estimator.AWS.upload_file"),
        mock_datetime,
    )


@pytest.fixture
def post_process_mocks(mocker: MockerFixture) -> PostProcessMocks:
    mock_datetime: MagicMock = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return PostProcessMocks(
        mocker.patch("finetune.hf_estimator.Path"),
        mocker.patch.object(HuggingFaceLoRATrainer, "download_output"),
        mocker.patch.object(HuggingFaceLoRATrainer, "merge"),
        mocker.patch.object(HuggingFaceLoRATrainer, "upload_output"),
        mock_datetime,
    )


def create_trainer(
    schema: type[BaseModel] = SchemaFixture,
    prompt_builder: BasePromptBuilder | None = None,
    training_batch_names: list[str] | None = None,
    hyperparameters: dict[str, str | int] | None = None,
    aws_config: dict[str, str] | None = None,
    model: str = "foo",
    description: str = "foo",
    instance_type: str = "foo.bar1.2baz",
    instance_count: int = 1,
    transformers_version: str = "1.23",
    pytorch_version: str = "1.2",
    py_version: str = "foo123",
) -> HuggingFaceLoRATrainerFixture:
    return HuggingFaceLoRATrainerFixture(
        schema,
        prompt_builder or PromptBuilderFixture(),
        training_batch_names or ["bar"],
        hyperparameters or {"base_model": "baz", "max_seq_length": 1024},
        aws_config or {"bucket": "qux", "region": "quux", "role": "corge"},
        model,
        description,
        instance_type,
        instance_count,
        transformers_version,
        pytorch_version,
        py_version,
    )


class TestConstructor:
    def test_init_sets_job_id_with_timestamp_and_description(
        self, constructor_mocks: ConstructorMocks
    ) -> None:
        assert (
            create_trainer(description="grault").get_job_id()
            == "20260101-120000-grault"
        )

    def test_init_sets_s3_input_path(self, constructor_mocks: ConstructorMocks) -> None:
        assert (
            create_trainer(description="plugh").get_s3_input_path()
            == "jobs/train/20260101-120000-plugh/input"
        )

    def test_init_sets_s3_output_path(
        self, constructor_mocks: ConstructorMocks
    ) -> None:
        assert (
            create_trainer(description="waldo").get_s3_output_path()
            == "jobs/train/20260101-120000-waldo/output"
        )

    def test_init_sets_s3_full_output_path(
        self, constructor_mocks: ConstructorMocks
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = create_trainer(
            aws_config={"bucket": "xyzzy", "region": "thud", "role": "wibble"},
            description="wobble",
        )
        assert (
            trainer.get_s3_full_output_path()
            == "s3://xyzzy/jobs/train/20260101-120000-wobble/output"
        )


class TestPrepareData:
    def test_prepare_data_calls_training_data_handler_prepare(
        self, prepare_data_mocks: PrepareDataMocks
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = create_trainer(
            training_batch_names=["20260101-120000_corge-quux"],
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "foo"},
        )
        trainer.prepare_data()
        prepare_data_mocks.training_data_handler.assert_called_once_with(
            schema=SchemaFixture,
            system_prompt="foo",
            training_batch_names=["20260101-120000_corge-quux"],
            base_model="baz",
            max_seq_length=1024,
            bucket="foo-bar",
            s3_prefix="trainingdata",
            output_file="train.jsonl",
            region="foo-bar-1",
            shuffle=True,
        )

    def test_prepare_data_calls_aws_upload_file(
        self, prepare_data_mocks: PrepareDataMocks
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = create_trainer(
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
        self, prepare_data_mocks: PrepareDataMocks
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = create_trainer(
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "baz"},
        )
        assert (
            trainer.prepare_data()
            == "s3://foo-bar/jobs/train/20260101-120000-foo/input"
        )

    def test_prepare_data_logs_job_id(
        self, prepare_data_mocks: PrepareDataMocks
    ) -> None:
        create_trainer(description="foo-bar").prepare_data()
        prepare_data_mocks.logger.info.assert_any_call(
            "Preparing training data for job: 20260101-120000-foo-bar"
        )

    def test_prepare_data_logs_s3_path(
        self, prepare_data_mocks: PrepareDataMocks
    ) -> None:
        create_trainer(description="foo-bar").prepare_data()
        prepare_data_mocks.logger.info.assert_any_call(
            "Uploading to S3: jobs/train/20260101-120000-foo-bar/input"
        )


class TestLaunchJob:
    def test_launch_job_creates_huggingface_estimator(
        self, launch_job_mocks: LaunchJobMocks
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = create_trainer(
            hyperparameters={"base_model": "foo-bar/Baz-1-2qux"},
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
            hyperparameters={"base_model": "foo-bar/Baz-1-2qux"},
        )

    def test_launch_job_calls_fit_with_training_path(
        self, launch_job_mocks: LaunchJobMocks
    ) -> None:
        create_trainer().launch_job("s3://bucket/path/to/data")
        launch_job_mocks.huggingface.return_value.fit.assert_called_once_with(
            {"training": "s3://bucket/path/to/data"}, wait=False
        )

    def test_launch_job_returns_job_name(
        self, launch_job_mocks: LaunchJobMocks
    ) -> None:
        assert create_trainer().launch_job("s3://bucket/input") == "mesa-foo-bar"

    def test_launch_job_logs_configuring_message(
        self, launch_job_mocks: LaunchJobMocks
    ) -> None:
        create_trainer().launch_job("s3://bucket/input")
        launch_job_mocks.logger.info.assert_any_call(
            "Configuring SageMaker HuggingFace estimator"
        )

    def test_launch_job_logs_launching_message(
        self, launch_job_mocks: LaunchJobMocks
    ) -> None:
        create_trainer().launch_job("s3://bucket/input")
        launch_job_mocks.logger.info.assert_any_call("Launching SageMaker training job")

    def test_launch_job_logs_job_name(self, launch_job_mocks: LaunchJobMocks) -> None:
        create_trainer().launch_job("s3://bucket/input")
        launch_job_mocks.logger.info.assert_any_call("Job launched: mesa-foo-bar")


class TestRun:
    def test_run_calls_prepare_data(self, run_mocks: RunMocks) -> None:
        create_trainer().run()
        run_mocks.prepare_data.assert_called_once()

    def test_run_calls_launch_job_with_prepare_data_result(
        self, run_mocks: RunMocks
    ) -> None:
        create_trainer().run()
        run_mocks.launch_job.assert_called_once_with("s3://foo/bar")

    def test_run_returns_job_name(self, run_mocks: RunMocks) -> None:
        assert create_trainer().run() == "mesa-foo-bar"

    def test_run_prints_starting_message(self, run_mocks: RunMocks) -> None:
        create_trainer(description="foo-bar").run()
        run_mocks.print.assert_any_call(
            "Starting training job: 20260101-120000-foo-bar"
        )

    def test_run_prints_preparing_data_message(self, run_mocks: RunMocks) -> None:
        create_trainer().run()
        run_mocks.print.assert_any_call("Preparing training data...")

    def test_run_prints_launching_message(self, run_mocks: RunMocks) -> None:
        create_trainer().run()
        run_mocks.print.assert_any_call("Launching SageMaker job...")

    def test_run_prints_job_launched_message(self, run_mocks: RunMocks) -> None:
        create_trainer().run()
        run_mocks.print.assert_any_call("Job launched: mesa-foo-bar")

    def test_run_sets_last_job_name(self, run_mocks: RunMocks) -> None:
        trainer: HuggingFaceLoRATrainerFixture = create_trainer()
        trainer.run()
        assert trainer.last_job_name == "mesa-foo-bar"


class TestDownloadOutput:
    def test_download_output_file_exists_returns_true(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = True
        assert create_trainer().download_output("foo/bar", "baz/qux", "quux")

    def test_download_output_file_exists_does_not_call_download(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = True
        create_trainer().download_output("foo/bar", "baz/qux", "quux")
        download_output_mocks.aws.assert_not_called()

    def test_download_output_file_exists_does_not_call_tarfile(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = True
        create_trainer().download_output("foo/bar", "baz/qux", "quux")
        download_output_mocks.tarfile.assert_not_called()

    def test_download_output_calls_aws_download_file(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = True
        trainer: HuggingFaceLoRATrainerFixture = create_trainer(
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

    def test_download_output_download_fails_raises_value_error(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = False
        with pytest.raises(ValueError, match="Failed to download training output"):
            create_trainer().download_output("foo/bar", "baz/qux", "quux")

    def test_download_output_download_fails_does_not_call_tarfile(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = False
        with pytest.raises(ValueError):
            create_trainer().download_output("foo/bar", "baz/qux", "quux")
        download_output_mocks.tarfile.assert_not_called()

    def test_download_output_success_extracts_tarfile(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = True
        create_trainer().download_output("foo/bar", "baz/qux", "quux")
        download_output_mocks.tarfile.assert_called_once_with(
            download_output_mocks.path.return_value, "r:*"
        )

    def test_download_output_success_extracts_to_parent_directory(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = True
        create_trainer().download_output("foo/bar", "baz/qux", "quux")
        download_output_mocks.tarfile.return_value.__enter__.return_value.extractall.assert_called_once_with(
            download_output_mocks.path.return_value.parent
        )

    def test_download_output_success_returns_true(
        self, download_output_mocks: DownloadOutputMocks
    ) -> None:
        download_output_mocks.path.return_value.exists.return_value = False
        download_output_mocks.aws.return_value = True
        assert create_trainer().download_output("foo/bar", "baz/qux", "quux")


class TestUploadOutput:
    @pytest.fixture
    def mock_model_card(self) -> MagicMock:
        mock: MagicMock = MagicMock()
        mock.model_name = "foo"
        mock.major = 1
        mock.minor = 2
        mock.patch = 3
        mock.to_yaml_bytes.return_value = b"bar"
        return mock

    def test_upload_output_archive_exists_does_not_create_tarfile(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        upload_output_mocks.aws.return_value = True
        create_trainer().upload_output("bar/baz", mock_model_card)
        upload_output_mocks.tarfile.assert_not_called()

    def test_upload_output_archive_exists_calls_aws_upload_file(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        upload_output_mocks.aws.return_value = True
        create_trainer(
            model="bar", aws_config={"bucket": "baz", "region": "qux", "role": "quux"}
        ).upload_output("grault/garply", mock_model_card, "waldo")
        upload_output_mocks.aws.assert_called_once_with(
            region_name="qux",
            file_name=str(
                upload_output_mocks.path.return_value.parent.__truediv__.return_value
            ),
            bucket="waldo",
            object_name="foo_1_2_3.tar.gz",
            path="bar",
        )

    def test_upload_output_archive_exists_upload_succeeds_returns_true(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        upload_output_mocks.aws.return_value = True
        assert create_trainer().upload_output("bar/baz", mock_model_card)

    def test_upload_output_archive_exists_upload_fails_raises_value_error(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        upload_output_mocks.aws.return_value = False
        with pytest.raises(ValueError, match="Failed to upload merged model weights"):
            create_trainer().upload_output("bar/baz", mock_model_card)

    def test_upload_output_archive_not_exists_creates_tarfile(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        upload_output_mocks.aws.return_value = True
        create_trainer().upload_output("bar/baz", mock_model_card)
        upload_output_mocks.tarfile.assert_called_once_with(
            upload_output_mocks.path.return_value.parent.__truediv__.return_value,
            "w:gz",
        )

    def test_upload_output_archive_not_exists_adds_all_items_from_target_path(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        upload_output_mocks.aws.return_value = True
        mock_item1: MagicMock = MagicMock()
        mock_item1.name = "baz.txt"
        mock_item2: MagicMock = MagicMock()
        mock_item2.name = "qux.txt"
        upload_output_mocks.path.return_value.iterdir.return_value = [
            mock_item1,
            mock_item2,
        ]
        create_trainer().upload_output("quux/corge", mock_model_card)
        mock_tar: MagicMock = (
            upload_output_mocks.tarfile.return_value.__enter__.return_value
        )
        assert mock_tar.add.call_count == 3
        mock_tar.add.assert_any_call(mock_item1, arcname="baz.txt")
        mock_tar.add.assert_any_call(mock_item2, arcname="qux.txt")

    def test_upload_output_archive_not_exists_adds_model_card_yaml(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        upload_output_mocks.aws.return_value = True
        upload_output_mocks.path.return_value.iterdir.return_value = []
        create_trainer().upload_output("baz/qux", mock_model_card)
        upload_output_mocks.tarinfo.assert_called_once_with(name="model_card.yml")
        upload_output_mocks.tarfile.return_value.__enter__.return_value.addfile.assert_called_once_with(
            upload_output_mocks.tarinfo.return_value,
            upload_output_mocks.io.return_value,
        )
        upload_output_mocks.io.assert_called_once_with(b"bar")

    def test_upload_output_archive_not_exists_adds_license(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        upload_output_mocks.aws.return_value = True
        upload_output_mocks.path.return_value.iterdir.return_value = []
        create_trainer().upload_output("baz/qux", mock_model_card)
        mock_tar: MagicMock = (
            upload_output_mocks.tarfile.return_value.__enter__.return_value
        )
        assert mock_tar.add.call_count == 1
        mock_tar.add.assert_called_once_with(
            upload_output_mocks.path.return_value.parents.__getitem__.return_value
            / "LICENSE.md",
            arcname="LICENSE.md",
        )

    def test_upload_output_archive_not_exists_upload_succeeds_returns_true(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        upload_output_mocks.aws.return_value = True
        upload_output_mocks.path.return_value.iterdir.return_value = []
        assert create_trainer().upload_output("baz/qux", mock_model_card)

    def test_upload_output_archive_not_exists_upload_fails_raises_value_error(
        self, upload_output_mocks: UploadOutputMocks, mock_model_card: MagicMock
    ) -> None:
        upload_output_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        upload_output_mocks.aws.return_value = False
        upload_output_mocks.path.return_value.iterdir.return_value = []
        with pytest.raises(ValueError, match="Failed to upload merged model weights"):
            create_trainer().upload_output("baz/qux", mock_model_card)


class TestPostProcess:
    @pytest.fixture
    def mock_model_card(self) -> MagicMock:
        mock: MagicMock = MagicMock()
        mock.model_name = "foo"
        mock.major = 1
        mock.minor = 2
        mock.patch = 3
        return mock

    def test_post_process_creates_source_folder(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        create_trainer(description="foo").post_process(
            mock_model_card, "bar/baz", "qux"
        )
        post_process_mocks.path.return_value.mkdir.assert_any_call(
            parents=True, exist_ok=True
        )

    def test_post_process_creates_target_folder(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        create_trainer(description="foo").post_process(
            mock_model_card, "bar/baz", "qux"
        )
        assert post_process_mocks.path.return_value.mkdir.call_count == 2

    def test_post_process_uses_provided_s3_output_path(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        create_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.download_output.assert_called_once()
        assert post_process_mocks.download_output.call_args[0][1] == "foo/bar"

    def test_post_process_uses_default_s3_output_path_when_none(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        trainer: HuggingFaceLoRATrainerFixture = create_trainer(description="foo")
        trainer.post_process(mock_model_card, None, "bar")
        post_process_mocks.download_output.assert_called_once()
        assert (
            post_process_mocks.download_output.call_args[0][1]
            == "jobs/train/20260101-120000-foo/output"
        )

    def test_post_process_uses_provided_job_name(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        create_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.download_output.assert_called_once()
        assert post_process_mocks.download_output.call_args[0][2] == "baz"

    def test_post_process_uses_last_job_name_when_none(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        trainer: HuggingFaceLoRATrainerFixture = create_trainer()
        trainer.last_job_name = "foo-bar"
        trainer.post_process(mock_model_card, "baz/qux", None)
        post_process_mocks.download_output.assert_called_once()
        assert post_process_mocks.download_output.call_args[0][2] == "foo-bar"

    def test_post_process_no_job_name_raises_value_error(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        trainer: HuggingFaceLoRATrainerFixture = create_trainer()
        trainer.last_job_name = None
        with pytest.raises(
            ValueError, match="no last job available and no job name specified"
        ):
            trainer.post_process(mock_model_card, "foo/bar", None)

    def test_post_process_download_fails_raises_value_error(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = False
        with pytest.raises(ValueError, match="downloading low-rank weights failed"):
            create_trainer().post_process(mock_model_card, "foo/bar", "baz")

    def test_post_process_download_fails_does_not_call_merge(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = False
        with pytest.raises(ValueError):
            create_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.merge.assert_not_called()

    def test_post_process_merge_fails_raises_value_error(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = False
        with pytest.raises(ValueError, match="merging with base model failed"):
            create_trainer().post_process(mock_model_card, "foo/bar", "baz")

    def test_post_process_merge_fails_does_not_call_upload(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = False
        with pytest.raises(ValueError):
            create_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.upload_output.assert_not_called()

    def test_post_process_calls_download_output(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        create_trainer(description="foo").post_process(
            mock_model_card, "bar/baz", "qux"
        )
        post_process_mocks.download_output.assert_called_once_with(
            str(post_process_mocks.path.return_value), "bar/baz", "qux"
        )

    def test_post_process_calls_merge(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        create_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.merge.assert_called_once_with(
            str(post_process_mocks.path.return_value),
            str(post_process_mocks.path.return_value),
        )

    def test_post_process_calls_upload_output(
        self, post_process_mocks: PostProcessMocks, mock_model_card: MagicMock
    ) -> None:
        post_process_mocks.download_output.return_value = True
        post_process_mocks.merge.return_value = True
        create_trainer().post_process(mock_model_card, "foo/bar", "baz")
        post_process_mocks.upload_output.assert_called_once_with(
            str(post_process_mocks.path.return_value), mock_model_card
        )
