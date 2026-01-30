from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, mock_open

import pytest
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from datagen.trainingdata_uploader import TrainingDataUploader
from tests.types import PathOperations


@dataclass
class FileOperations:
    open: MagicMock
    remove: MagicMock


@dataclass
class TarOperations:
    tarfile_open: MagicMock


@dataclass
class UploadDependencies:
    get_schema_version: MagicMock
    upload_file: MagicMock
    model_validate_json: MagicMock
    datetime: MagicMock
    os_remove: MagicMock


@dataclass
class HelperMethods:
    create_jsonl: MagicMock
    create_metadata: MagicMock
    create_samples_archive: MagicMock


@pytest.fixture
def mock_file_operations(mocker: MockerFixture) -> FileOperations:
    return FileOperations(
        open=mocker.patch("builtins.open", mock_open()),
        remove=mocker.patch("os.remove", autospec=True),
    )


@pytest.fixture
def mock_tar_operations(mocker: MockerFixture) -> TarOperations:
    mock_tar: MagicMock = MagicMock()
    mock_tar.__enter__ = MagicMock(return_value=mock_tar)
    mock_tar.__exit__ = MagicMock(return_value=False)
    return TarOperations(
        tarfile_open=mocker.patch("tarfile.open", autospec=True, return_value=mock_tar),
    )


@pytest.fixture
def mock_upload_dependencies(mocker: MockerFixture) -> UploadDependencies:
    return UploadDependencies(
        get_schema_version=mocker.patch(
            "datagen.trainingdata_uploader.get_schema_version",
            autospec=True,
            return_value="1_2_3",
        ),
        upload_file=mocker.patch(
            "datagen.trainingdata_uploader.AWS.upload_file", autospec=True
        ),
        model_validate_json=mocker.patch(
            "datagen.trainingdata_uploader.TrainingExample.model_validate_json",
            autospec=True,
        ),
        datetime=mocker.patch("datagen.trainingdata_uploader.datetime", autospec=True),
        os_remove=mocker.patch(
            "datagen.trainingdata_uploader.os.remove", autospec=True
        ),
    )


@pytest.fixture
def mock_helper_methods(mocker: MockerFixture) -> HelperMethods:
    return HelperMethods(
        create_jsonl=mocker.patch.object(
            TrainingDataUploader, "_create_jsonl", return_value="test.jsonl"
        ),
        create_metadata=mocker.patch.object(
            TrainingDataUploader, "_create_metadata", return_value="metadata.yaml"
        ),
        create_samples_archive=mocker.patch.object(
            TrainingDataUploader,
            "_create_samples_archive",
            return_value="samples.tar.gz",
        ),
    )


@pytest.fixture
def mock_uploader_path_operations(mocker: MockerFixture) -> PathOperations:
    mock_glob: MagicMock = mocker.patch.object(Path, "glob")
    mock_glob.return_value = [
        Path(f"./data/trainingdata/sample_{i}.json") for i in range(3)
    ]
    return PathOperations(
        read_text=mocker.patch.object(Path, "read_text", return_value="{}"),
        glob=mock_glob,
        exists=mocker.patch.object(Path, "exists", return_value=True),
    )


@pytest.fixture
def mock_schema() -> type[BaseModel]:
    class MockSchema(BaseModel):
        field: str

    return MockSchema


def test_create_jsonl_valid_samples_writes_correct_format(
    mock_file_operations: FileOperations,
) -> None:
    samples: list[dict[str, object]] = [
        {"content": "foo", "output": {"field": "bar"}},
        {"content": "baz", "output": {"field": "qux"}},
    ]
    assert (
        TrainingDataUploader._create_jsonl(samples, "foo", "bar")
        == "train_openai_bar.jsonl"
    )
    mock_file_operations.open.assert_called_once_with("train_openai_bar.jsonl", "w")


def test_create_jsonl_empty_samples_creates_empty_file(
    mock_file_operations: FileOperations,
) -> None:
    assert (
        TrainingDataUploader._create_jsonl([], "foo", "empty")
        == "train_openai_empty.jsonl"
    )
    handle = mock_file_operations.open()
    handle.write.assert_not_called()


def test_create_metadata_valid_params_writes_yaml(
    mocker: MockerFixture,
    mock_file_operations: FileOperations,
) -> None:
    mock_yaml_dump: MagicMock = mocker.patch(
        "datagen.trainingdata_uploader.yaml.dump", autospec=True
    )
    mock_datetime: MagicMock = mocker.patch(
        "datagen.trainingdata_uploader.datetime", autospec=True
    )
    mock_datetime.now.return_value.isoformat.return_value = "2026-01-01T00:00:00"
    schema: str = "foo_schema"
    schema_version: str = "1_2_3"
    short_description: str = "foo"
    description: str = "foobar"
    num_samples: int = 10
    assert (
        TrainingDataUploader._create_metadata(
            schema, schema_version, short_description, description, num_samples
        )
        == "metadata.yaml"
    )
    mock_yaml_dump.assert_called_once()
    call_args: dict[str, str | int] = mock_yaml_dump.call_args[0][0]
    assert call_args["schema"] == schema
    assert call_args["schema_version"] == schema_version
    assert call_args["short_description"] == short_description
    assert call_args["description"] == description
    assert call_args["num_samples"] == num_samples


def test_create_samples_archive_valid_files_creates_tar(
    mock_tar_operations: TarOperations,
) -> None:
    assert (
        TrainingDataUploader._create_samples_archive(
            [Path("foo.json"), Path("bar.json")], "test"
        )
        == "train_openai_test_samples.tar.gz"
    )
    mock_tar_operations.tarfile_open.assert_called_once_with(
        "train_openai_test_samples.tar.gz", "w:gz"
    )


def test_create_samples_archive_adds_all_files_to_archive(
    mock_tar_operations: TarOperations,
) -> None:
    TrainingDataUploader._create_samples_archive(
        [Path("foo.json"), Path("bar.json"), Path("baz.json")], "test"
    )
    tar_context = mock_tar_operations.tarfile_open.return_value.__enter__.return_value
    assert tar_context.add.call_count == 3


def test_upload_folder_not_exists_raises_value_error(
    mocker: MockerFixture,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mocker.patch.object(Path, "exists", return_value=False)
    with pytest.raises(ValueError, match="Input folder does not exist"):
        TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")


def test_upload_no_valid_samples_raises_value_error(
    mocker: MockerFixture,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_uploader_path_operations.glob.return_value = []
    with pytest.raises(ValueError, match="No valid samples found"):
        TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")


def test_upload_validation_error_skips_sample(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.side_effect = [
        ValidationError.from_exception_data("error", []),
        mock_example,
        mock_example,
    ]
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")
    assert mock_upload_dependencies.upload_file.call_count == 3


def test_upload_generic_exception_skips_sample(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.side_effect = [
        Exception("load error"),
        mock_example,
        mock_example,
    ]
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")
    assert mock_upload_dependencies.upload_file.call_count == 3


def test_upload_valid_samples_uploads_three_files(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.return_value = mock_example
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")
    assert mock_upload_dependencies.upload_file.call_count == 3


def test_upload_valid_samples_removes_three_files(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.return_value = mock_example
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")
    assert mock_upload_dependencies.os_remove.call_count == 3


def test_upload_returns_correct_s3_uri(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.return_value = mock_example
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    assert (
        TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")
        == "s3://aicentre-nlpteam-mesa-build/trainingdata/20260101-000000_foo/"
    )


def test_upload_custom_bucket_uses_custom_bucket(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.return_value = mock_example
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    result: str = TrainingDataUploader.upload(
        mock_schema, "foo_schema", "bar", "foo", bucket="baz"
    )
    assert result == "s3://baz/trainingdata/20260101-000000_foo/"


def test_upload_long_description_none_uses_short_description(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.return_value = mock_example
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")
    mock_helper_methods.create_metadata.assert_called_once_with(
        "foo_schema", "1_2_3", "foo", "foo", 3
    )


def test_upload_long_description_provided_uses_long_description(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
    mock_schema: type[BaseModel],
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.return_value = mock_example
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    TrainingDataUploader.upload(
        mock_schema,
        "foo_schema",
        "bar",
        "foo",
        long_description="foobar",
    )
    mock_helper_methods.create_metadata.assert_called_once_with(
        "foo_schema", "1_2_3", "foo", "foobar", 3
    )


def test_upload_schema_validation_error_skips_sample(
    mock_helper_methods: HelperMethods,
    mock_uploader_path_operations: PathOperations,
    mock_upload_dependencies: UploadDependencies,
) -> None:
    mock_example: MagicMock = MagicMock()
    mock_example.content = "foo"
    mock_example.output = {"field": "bar"}
    mock_upload_dependencies.model_validate_json.return_value = mock_example
    mock_upload_dependencies.datetime.now.return_value.strftime.return_value = (
        "20260101-000000"
    )
    mock_schema: MagicMock = MagicMock()
    mock_schema.model_validate.side_effect = [
        ValidationError.from_exception_data("error", []),
        None,
        None,
    ]
    TrainingDataUploader.upload(mock_schema, "foo_schema", "bar", "foo")
    assert mock_upload_dependencies.upload_file.call_count == 3
