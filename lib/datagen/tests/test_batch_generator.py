import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open

import pytest
from pytest_mock import MockerFixture

from datagen.batch_generator import BatchState, BedrockBatchGenerator
from mesa_types import Document
from tests.types import PathOperations


class BatchGeneratorFixture(BedrockBatchGenerator):
    def generate_batch(
        self, sample_size: int, file_name: str = "foo.jsonl"
    ) -> list[str]:
        return self._generate_batch(sample_size, file_name)

    def documents_by_record_id(self) -> dict[str, Document]:
        return self._documents_by_record_id()

    def read_state(self) -> BatchState | None:
        return self._read_state()

    def write_state(self, state: BatchState) -> None:
        self._write_state(state)

    def pending_documents(self, sample_size: int) -> dict[str, Document]:
        return self._pending_documents(sample_size)


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
    _pending_documents: MagicMock
    create_anthropic_bedrock_batch_entry: MagicMock


@dataclass
class PendingDependencies:
    _documents_by_record_id: MagicMock
    _read_state: MagicMock


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
    _read_state: MagicMock
    _write_state: MagicMock
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
def mock_batch_dependencies(
    mocker: MockerFixture, generator: BatchGeneratorFixture
) -> BatchDependencies:
    return BatchDependencies(
        _pending_documents=mocker.patch.object(
            generator,
            "_pending_documents",
            autospec=True,
            return_value={
                f"id{index}": MagicMock(source="foo", content=f"bar{index}")
                for index in range(3)
            },
        ),
        create_anthropic_bedrock_batch_entry=mocker.patch(
            "datagen.batch_generator.AWS.create_anthropic_bedrock_batch_entry",
            autospec=True,
            return_value={"recordId": "0", "modelInput": {}},
        ),
    )


@pytest.fixture
def mock_pending_dependencies(
    mocker: MockerFixture, generator: BatchGeneratorFixture
) -> PendingDependencies:
    return PendingDependencies(
        _documents_by_record_id=mocker.patch.object(
            generator,
            "_documents_by_record_id",
            autospec=True,
            return_value={f"id{index}": MagicMock() for index in range(5)},
        ),
        _read_state=mocker.patch.object(
            generator, "_read_state", autospec=True, return_value=None
        ),
    )


@pytest.fixture
def mock_extract_dependencies(mocker: MockerFixture) -> ExtractDependencies:
    mock_output: MagicMock = MagicMock()
    mock_output.modelOutput.content = [MagicMock(text="sample_text")]
    mock_output.recordId = hashlib.md5(b"bar").hexdigest()
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
            generator, "_generate_batch", autospec=True, return_value=["foo", "bar"]
        ),
        _read_state=mocker.patch.object(
            generator, "_read_state", autospec=True, return_value=None
        ),
        _write_state=mocker.patch.object(generator, "_write_state", autospec=True),
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


def test_init_multiple_batches_interleaves_document_files(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_generator_dependencies: GeneratorDependencies,
) -> None:
    mocker.patch.object(
        Path,
        "glob",
        side_effect=[
            [Path("./data/documents/batch1/foo.json")],
            [Path("./data/documents/batch2/bar.json")],
        ],
    )
    mock_interleave: MagicMock = mocker.patch(
        "datagen.batch_generator.DocumentLoader.interleave", autospec=True
    )
    BatchGeneratorFixture(
        system_prompt="foo",
        schema=MagicMock,
        schema_name="baz",
        model_name="foo_model",
        document_batches=["batch1.tar.gz", "batch2.tar"],
    )
    mock_interleave.assert_called_once_with(
        [
            [Path("./data/documents/batch1/foo.json")],
            [Path("./data/documents/batch2/bar.json")],
        ]
    )


def test_documents_by_record_id_documents_read_keys_by_content_hash(
    mocker: MockerFixture,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    generator: BatchGeneratorFixture,
) -> None:
    documents: list[MagicMock] = [
        MagicMock(source="foo", content=f"bar{index}") for index in range(5)
    ]
    mocker.patch(
        "datagen.batch_generator.Document.model_validate_json", side_effect=documents
    )
    assert generator.documents_by_record_id() == {
        hashlib.md5(document.content.encode()).hexdigest(): document
        for document in documents
    }


def test_documents_by_record_id_duplicate_content_keeps_one_entry(
    mocker: MockerFixture,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch(
        "datagen.batch_generator.Document.model_validate_json",
        return_value=MagicMock(source="foo", content="bar"),
    )
    assert list(generator.documents_by_record_id()) == [hashlib.md5(b"bar").hexdigest()]


def test_read_state_file_absent_returns_none(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch.object(Path, "exists", return_value=False)
    assert generator.read_state() is None


def test_read_state_file_present_returns_state(
    mocker: MockerFixture,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch(
        "builtins.open",
        mock_open(read_data='{"job_id": "foo/bar", "sent_record_ids": ["baz"]}'),
    )
    assert generator.read_state() == BatchState(
        job_id="foo/bar", sent_record_ids=["baz"]
    )


def test_read_state_file_without_sent_records_returns_empty_record_ids(
    mocker: MockerFixture,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch("builtins.open", mock_open(read_data='{"job_id": "foo/bar"}'))
    assert generator.read_state() == BatchState(job_id="foo/bar")


def test_write_state_state_given_writes_json(
    mock_filesystem: FileSystem,
    generator: BatchGeneratorFixture,
) -> None:
    generator.write_state(BatchState(job_id="foo/bar", sent_record_ids=["baz"]))
    mock_filesystem.open.assert_called_once_with(".job_id.json", "w")
    mock_filesystem.open().write.assert_called_once_with(
        '{"job_id":"foo/bar","sent_record_ids":["baz"]}'
    )


@pytest.mark.parametrize(
    "state, expected",
    [
        (None, ["id0", "id1", "id2"]),
        (BatchState(job_id="foo/bar"), ["id0", "id1", "id2"]),
        (
            BatchState(job_id="foo/bar", sent_record_ids=["id0", "id1", "id2"]),
            ["id3", "id4"],
        ),
    ],
)
def test_pending_documents_documents_sent_returns_next_unsent_documents(
    mock_filesystem: FileSystem,
    mock_pending_dependencies: PendingDependencies,
    generator: BatchGeneratorFixture,
    state: BatchState | None,
    expected: list[str],
) -> None:
    mock_pending_dependencies._read_state.return_value = state
    assert list(generator.pending_documents(3)) == expected


def test_pending_documents_fewer_remain_than_requested_warns(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_pending_dependencies: PendingDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_logger: MagicMock = mocker.patch.object(generator, "_logger")
    mock_pending_dependencies._read_state.return_value = BatchState(
        job_id="foo/bar", sent_record_ids=["id0", "id1", "id2"]
    )
    generator.pending_documents(3)
    mock_logger.warning.assert_called_once()


def test_pending_documents_all_documents_sent_raises_value_error(
    mock_filesystem: FileSystem,
    mock_pending_dependencies: PendingDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_pending_dependencies._read_state.return_value = BatchState(
        job_id="foo/bar", sent_record_ids=[f"id{index}" for index in range(5)]
    )
    with pytest.raises(ValueError, match="already been sent"):
        generator.pending_documents(3)


def test_generate_batch_sample_size_given_creates_correct_entries(
    mock_filesystem: FileSystem,
    mock_batch_dependencies: BatchDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    generator.generate_batch(3)
    mock_batch_dependencies._pending_documents.assert_called_once_with(3)
    assert [
        call.args[0]
        for call in mock_batch_dependencies.create_anthropic_bedrock_batch_entry.call_args_list
    ] == ["id0", "id1", "id2"]


def test_generate_batch_documents_pending_returns_record_ids(
    mock_filesystem: FileSystem,
    mock_batch_dependencies: BatchDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    assert generator.generate_batch(3, "custom.jsonl") == ["id0", "id1", "id2"]
    mock_filesystem.open.assert_called_once_with("custom.jsonl", "w")


def test_generate_via_batch_valid_params_calls_generate_inference_writes_file_returns_id(
    mock_filesystem: FileSystem,
    mock_generate_via_batch_dependencies: GenerateViaBatchDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_generate_via_batch_dependencies.datetime.now.return_value.strftime.return_value = "2026-01-01-0000"
    result: str = generator.generate_via_batch(10, "bucket", "role_arn")
    mock_generate_via_batch_dependencies._generate_batch.assert_called_once_with(10)
    mock_generate_via_batch_dependencies.run_batch_inference.assert_called_once()
    assert result == "datagen/2026-01-01-0000"


@pytest.mark.parametrize(
    "previous, expected",
    [
        (None, ["foo", "bar"]),
        (
            BatchState(job_id="datagen/2025-01-01-0000", sent_record_ids=["baz"]),
            ["baz", "foo", "bar"],
        ),
    ],
)
def test_generate_via_batch_batch_submitted_appends_sent_record_ids(
    mock_filesystem: FileSystem,
    mock_generate_via_batch_dependencies: GenerateViaBatchDependencies,
    generator: BatchGeneratorFixture,
    previous: BatchState | None,
    expected: list[str],
) -> None:
    mock_generate_via_batch_dependencies.datetime.now.return_value.strftime.return_value = "2026-01-01-0000"
    mock_generate_via_batch_dependencies._read_state.return_value = previous
    generator.generate_via_batch(10, "bucket", "role_arn")
    mock_generate_via_batch_dependencies._write_state.assert_called_once_with(
        BatchState(job_id="datagen/2026-01-01-0000", sent_record_ids=expected)
    )


def test_generate_via_batch_submission_fails_records_nothing(
    mock_filesystem: FileSystem,
    mock_generate_via_batch_dependencies: GenerateViaBatchDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mock_generate_via_batch_dependencies.run_batch_inference.side_effect = ValueError(
        "Error submitting job"
    )
    with pytest.raises(ValueError, match="Error submitting job"):
        generator.generate_via_batch(10, "bucket", "role_arn")
    mock_generate_via_batch_dependencies._write_state.assert_not_called()


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


def test_extract_batch_output_record_id_unknown_increments_failed_count(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    mocker.patch.object(generator, "_logger")
    mock_extract_dependencies.parse_batch_output.return_value = MagicMock(
        outputs=[MagicMock(recordId="foo")]
    )
    assert generator.extract_batch_output() == (0, 1)


def test_extract_batch_output_records_out_of_order_pairs_own_document(
    mock_filesystem: FileSystem,
    mock_extract_dependencies: ExtractDependencies,
    generator: BatchGeneratorFixture,
) -> None:
    documents: list[MagicMock] = [
        MagicMock(source="foo", content=f"bar{index}") for index in range(5)
    ]
    mock_extract_dependencies.model_validate_json.side_effect = documents
    mock_extract_dependencies.parse_batch_output.return_value = MagicMock(
        outputs=[
            MagicMock(recordId=hashlib.md5(document.content.encode()).hexdigest())
            for document in reversed(documents)
        ]
    )
    generator.extract_batch_output()
    assert [
        call.args[2]
        for call in mock_extract_dependencies.save_training_sample.call_args_list
    ] == [document.content for document in reversed(documents)]


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
