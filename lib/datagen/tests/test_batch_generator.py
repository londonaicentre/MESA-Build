from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open

import pytest
from pytest_mock import MockerFixture

from datagen.batch_generator import BedrockBatchGenerator
from tests.types import PathOperations


class BatchGeneratorFixture(BedrockBatchGenerator):
    def generate_batch(self, sample_size: int, file_name: str = "foo.jsonl") -> str:
        return self._generate_batch(sample_size, file_name)


@dataclass
class MockConfig:
    models: dict[str, Any]
    job_id_file: str = ".job_id.json"


@dataclass
class FileSystem:
    makedirs: MagicMock
    environ: MagicMock
    open: MagicMock


@dataclass
class GeneratorDependencies:
    config: MagicMock
    download_and_extract: MagicMock
    get_schema_version: MagicMock


@dataclass
class BatchDependencies:
    model_validate_json: MagicMock
    create_anthropic_bedrock_batch_entry: MagicMock


@dataclass
class ExtractDependencies:
    get_batch_inference_outputs: MagicMock
    parse_batch_output: MagicMock
    model_validate_json: MagicMock
    save_training_sample: MagicMock


@dataclass
class GenerateViaBatchDependencies:
    run_batch_inference: MagicMock
    _generate_batch: MagicMock
    datetime: MagicMock


@dataclass
class CheckStatusDependencies:
    list_s3_objects: MagicMock


@pytest.fixture
def mock_filesystem(mocker: MockerFixture) -> FileSystem:
    return FileSystem(
        makedirs=mocker.patch("os.makedirs", autospec=True),
        environ=mocker.patch.dict("os.environ", {}, clear=False),
        open=mocker.patch("builtins.open", mock_open()),
    )


@pytest.fixture
def mock_generator_dependencies(mocker: MockerFixture) -> GeneratorDependencies:
    mock_config: MagicMock = mocker.patch(
        "datagen.batch_generator.Config", autospec=True
    )
    mock_config.return_value = MockConfig(
        models={
            "foo_model": MagicMock(
                model="bedrock/foo", region="eu-west-2", batch_file="batch.jsonl"
            )
        }
    )
    return GeneratorDependencies(
        config=mock_config,
        download_and_extract=mocker.patch(
            "datagen.batch_generator.DocumentLoader.download_and_extract", autospec=True
        ),
        get_schema_version=mocker.patch(
            "datagen.batch_generator.get_schema_version",
            autospec=True,
            return_value="1_2_3",
        ),
    )


@pytest.fixture
def generator(
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_generator_dependencies: GeneratorDependencies,
) -> BatchGeneratorFixture:
    return BatchGeneratorFixture(
        system_prompt="foo",
        schema=MagicMock,
        schema_name="baz",
        model_name="foo_model",
        document_batches=["foobar.tar.gz"],
    )


@pytest.fixture
def mock_batch_dependencies(mocker: MockerFixture) -> BatchDependencies:
    return BatchDependencies(
        model_validate_json=mocker.patch(
            "datagen.batch_generator.Document.model_validate_json", autospec=True
        ),
        create_anthropic_bedrock_batch_entry=mocker.patch(
            "datagen.batch_generator.AWS.create_anthropic_bedrock_batch_entry",
            autospec=True,
            return_value={"recordId": "0", "modelInput": {}},
        ),
    )


@pytest.fixture
def mock_extract_dependencies(mocker: MockerFixture) -> ExtractDependencies:
    mock_output: MagicMock = MagicMock()
    mock_output.modelOutput.content = [MagicMock(text="sample_text")]
    mock_batch_outputs: MagicMock = MagicMock(outputs=[mock_output] * 3)
    return ExtractDependencies(
        get_batch_inference_outputs=mocker.patch(
            "datagen.batch_generator.AWS.get_batch_inference_outputs",
            autospec=True,
            return_value=mock_batch_outputs,
        ),
        parse_batch_output=mocker.patch(
            "datagen.batch_generator.AWS.parse_batch_output",
            autospec=True,
            return_value=mock_batch_outputs,
        ),
        model_validate_json=mocker.patch(
            "datagen.batch_generator.Document.model_validate_json",
            autospec=True,
            return_value=MagicMock(source="foo", content="bar"),
        ),
        save_training_sample=mocker.patch(
            "datagen.batch_generator.save_training_sample",
            autospec=True,
            return_value=True,
        ),
    )


@pytest.fixture
def mock_generate_via_batch_dependencies(
    mocker: MockerFixture, generator: BatchGeneratorFixture
) -> GenerateViaBatchDependencies:
    return GenerateViaBatchDependencies(
        run_batch_inference=mocker.patch(
            "datagen.batch_generator.AWS.run_batch_inference", autospec=True
        ),
        _generate_batch=mocker.patch.object(
            generator, "_generate_batch", autospec=True, return_value="batch.jsonl"
        ),
        datetime=mocker.patch(
            "datagen.batch_generator.datetime",
            autospec=True,
        ),
    )


def test_init_batches_provided_downloads_and_extracts(
    mock_generator_dependencies: GeneratorDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_generator_dependencies.download_and_extract.assert_called_once_with(
        filename="foobar.tar.gz", output_folder=Path("./data/documents/foobar")
    )


def test_init_model_with_region_sets_aws_region_env_var(
    mock_filesystem: FileSystem,
    generator: BatchGeneratorFixture,
) -> None:
    assert "AWS_REGION_NAME" in mock_filesystem.environ


def test_init_multiple_batches_downloads_all(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_generator_dependencies: GeneratorDependencies,
) -> None:
    mock_glob: MagicMock = mocker.patch.object(Path, "glob")
    mock_glob.return_value = [
        Path(f"./data/documents/foo/document_{i}.json") for i in range(2)
    ]
    BatchGeneratorFixture(
        system_prompt="foo",
        schema=MagicMock,
        schema_name="baz",
        model_name="foo_model",
        document_batches=["batch1.tar.gz", "batch2.tar"],
    )
    assert mock_generator_dependencies.download_and_extract.call_count == 2


def test_generate_batch_sample_size_given_creates_correct_entries(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_batch_dependencies: BatchDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_batch_dependencies.model_validate_json.return_value = MagicMock(
        source="foo", content="bar"
    )
    generator.generate_batch(3)
    assert mock_batch_dependencies.create_anthropic_bedrock_batch_entry.call_count == 3


def test_generate_batch_sample_exceeds_docs_caps_at_available(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_batch_dependencies: BatchDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_batch_dependencies.model_validate_json.return_value = MagicMock(
        source="foo", content="bar"
    )
    generator.generate_batch(10)
    assert mock_batch_dependencies.create_anthropic_bedrock_batch_entry.call_count == 5


def test_generate_batch_custom_filename_returns_filename(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_batch_dependencies: BatchDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_batch_dependencies.model_validate_json.return_value = MagicMock(
        source="foo", content="bar"
    )
    assert generator.generate_batch(1, "custom.jsonl") == "custom.jsonl"


def test_generate_via_batch_valid_params_calls_generate_inference_writes_file_returns_id(
    mock_filesystem: FileSystem,
    mock_generate_via_batch_dependencies: GenerateViaBatchDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_generate_via_batch_dependencies.datetime.now.return_value.strftime.return_value = "2026-01-01-0000"
    result: str = generator.generate_via_batch(10, "bucket", "role_arn")
    mock_generate_via_batch_dependencies._generate_batch.assert_called_once_with(10)
    mock_generate_via_batch_dependencies.run_batch_inference.assert_called_once()
    mock_filesystem.open.assert_called()
    assert result == "datagen/2026-01-01-0000"


@pytest.fixture
def mock_check_status_dependencies(mocker: MockerFixture) -> CheckStatusDependencies:
    return CheckStatusDependencies(
        list_s3_objects=mocker.patch(
            "datagen.batch_generator.AWS.list_s3_objects", autospec=True
        )
    )


def test_extract_batch_output_no_bucket_no_download(
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    generator.extract_batch_output()
    mock_extract_dependencies.get_batch_inference_outputs.assert_not_called()


def test_extract_batch_output_bucket_provided_downloads_file(
    mocker: MockerFixture,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch("builtins.open", mock_open(read_data='{"job_id": "foo/bar"}'))
    generator.extract_batch_output("test-bucket")
    mock_extract_dependencies.get_batch_inference_outputs.assert_called_once()


def test_extract_batch_output_download_fails_raises_value_error(
    mocker: MockerFixture,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch("builtins.open", mock_open(read_data='{"job_id": "foo/bar"}'))
    mock_extract_dependencies.get_batch_inference_outputs.side_effect = ValueError(
        "Error downloading file"
    )
    with pytest.raises(ValueError, match="Error downloading file"):
        generator.extract_batch_output("test-bucket")


@pytest.mark.parametrize(
    "side_effect,expected",
    [
        (None, (3, 0)),
        ([True, False, True], (2, 1)),
    ],
)
def test_extract_batch_output_returns_correct_count(
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
    side_effect: list[bool] | None,
    expected: tuple[int, int],
) -> None:
    if side_effect:
        mock_extract_dependencies.save_training_sample.side_effect = side_effect
    successful, failed = generator.extract_batch_output()
    assert (successful, failed) == expected


def test_extract_batch_output_exception_raised_increments_failed_count(
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_extract_dependencies.model_validate_json.side_effect = [
        MagicMock(source="foo", content="bar"),
        Exception("error"),
        MagicMock(source="foo", content="bar"),
    ]
    successful, failed = generator.extract_batch_output()
    assert successful == 2
    assert failed == 1


def test_extract_batch_output_called_creates_output_directory(
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    generator.extract_batch_output()
    mock_filesystem.makedirs.assert_called_with("./data/trainingdata/", exist_ok=True)


def test_check_batch_output_status_output_present_returns_true(
    mocker: MockerFixture,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_check_status_dependencies: CheckStatusDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch("builtins.open", mock_open(read_data='{"job_id": "foo/bar"}'))
    mock_check_status_dependencies.list_s3_objects.return_value = [
        {"Key": "foo/bar/output/batch.jsonl.out"}
    ]
    assert generator.check_batch_output_status("test-bucket") is True


def test_check_batch_output_status_output_absent_returns_false(
    mocker: MockerFixture,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_check_status_dependencies: CheckStatusDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch("builtins.open", mock_open(read_data='{"job_id": "foo/bar"}'))
    mock_check_status_dependencies.list_s3_objects.return_value = [
        {"Key": "foo/bar/output/other.jsonl.out"}
    ]
    assert generator.check_batch_output_status("test-bucket") is False


def test_check_batch_output_status_no_job_id_raises_value_error(
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_check_status_dependencies: CheckStatusDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_path_operations.exists.return_value = False
    with pytest.raises(ValueError, match="No batch job id found"):
        generator.check_batch_output_status("test-bucket")


def test_extract_batch_output_job_id_missing_skips_download(
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_path_operations.exists.return_value = False
    generator.extract_batch_output("test-bucket")
    mock_extract_dependencies.get_batch_inference_outputs.assert_not_called()
