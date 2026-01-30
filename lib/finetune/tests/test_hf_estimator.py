from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from finetune.hf_estimator import HuggingFaceLoRATrainer
from utils.prompt import BasePromptBuilder

from conftest import SchemaFixture


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


@pytest.fixture
def constructor_mocks(mocker: MockerFixture) -> ConstructorMocks:
    mock_datetime = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    return ConstructorMocks(mock_datetime)


@pytest.fixture
def prepare_data_mocks(mocker: MockerFixture) -> PrepareDataMocks:
    mock_datetime = mocker.patch("finetune.hf_estimator.datetime")
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
    mock_datetime = mocker.patch("finetune.hf_estimator.datetime")
    mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
    mock_hf = mocker.patch("finetune.hf_estimator.HuggingFace")
    mock_hf.return_value.latest_training_job.name = "mesa-foo-bar"
    mock_path = mocker.patch("finetune.hf_estimator.Path")
    mock_path.return_value.parent.__truediv__.return_value = "/foo/scripts"
    return LaunchJobMocks(
        mock_hf,
        mock_path,
        mocker.patch("finetune.hf_estimator.logger"),
        mock_datetime,
    )


@pytest.fixture
def run_mocks(mocker: MockerFixture) -> RunMocks:
    mock_datetime = mocker.patch("finetune.hf_estimator.datetime")
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


def create_trainer(
    schema: type[BaseModel] = SchemaFixture,
    prompt_builder: BasePromptBuilder | None = None,
    training_batch_names: list[str] | None = None,
    hyperparameters: dict[str, str] | None = None,
    aws_config: dict[str, str] | None = None,
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
        hyperparameters or {"base_model": "baz"},
        aws_config or {"bucket": "qux", "region": "quux", "role": "corge"},
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

    def test_init_sets_s3_output_path_with_full_uri(
        self, constructor_mocks: ConstructorMocks
    ) -> None:
        trainer = create_trainer(
            aws_config={"bucket": "xyzzy", "region": "thud", "role": "wibble"},
            description="wobble",
        )
        assert (
            trainer.get_s3_output_path()
            == "s3://xyzzy/jobs/train/20260101-120000-wobble/output"
        )


class TestPrepareData:
    def test_prepare_data_calls_training_data_handler_prepare(
        self, prepare_data_mocks: PrepareDataMocks
    ) -> None:
        trainer = create_trainer(
            training_batch_names=["20260101-120000_corge-quux"],
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "foo"},
        )
        trainer.prepare_data()
        prepare_data_mocks.training_data_handler.assert_called_once_with(
            schema=SchemaFixture,
            system_prompt="foo",
            training_batch_names=["20260101-120000_corge-quux"],
            bucket="foo-bar",
            s3_prefix="trainingdata",
            output_file="train.jsonl",
            region="foo-bar-1",
            shuffle=True,
        )

    def test_prepare_data_calls_aws_upload_file(
        self, prepare_data_mocks: PrepareDataMocks
    ) -> None:
        trainer = create_trainer(
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
        trainer = create_trainer(
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
        trainer = create_trainer(
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
