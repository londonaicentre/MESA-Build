from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from datagen.llm_generator import LLMGenerator
from tests.test_batch_generator import PathOperations


class GeneratorFixture(LLMGenerator):
    def generate_sample(self, doc_path: Path) -> bool:
        return self._generate_sample(doc_path)

    def get_logger(self) -> Logger:
        return self._logger


@dataclass
class FileSystem:
    exists: MagicMock
    makedirs: MagicMock


@dataclass
class GeneratorDependencies:
    get_output_filename: MagicMock
    download_and_extract: MagicMock
    get_schema_version: MagicMock


@dataclass
class SampleDependencies:
    model_validate_json: MagicMock
    completion: MagicMock
    save_training_sample: MagicMock


@pytest.fixture
def mock_filesystem(mocker: MockerFixture) -> FileSystem:
    return FileSystem(
        exists=mocker.patch("os.path.exists", autospec=True, return_value=False),
        makedirs=mocker.patch("os.makedirs", autospec=True),
    )


@pytest.fixture
def mock_generator_dependencies(mocker: MockerFixture) -> GeneratorDependencies:
    return GeneratorDependencies(
        get_output_filename=mocker.patch(
            "datagen.llm_generator.get_output_filename",
            autospec=True,
            return_value="baz.json",
        ),
        download_and_extract=mocker.patch(
            "datagen.llm_generator.DocumentLoader.download_and_extract", autospec=True
        ),
        get_schema_version=mocker.patch(
            "datagen.llm_generator.get_schema_version",
            autospec=True,
            return_value="1_2_3",
        ),
    )


@pytest.fixture
def generator(
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_generator_dependencies: GeneratorDependencies,
) -> GeneratorFixture:
    return GeneratorFixture(
        system_prompt="foo",
        schema=MagicMock,
        schema_name="baz",
        model_name="qux",
        api_key="quux",
        document_batches=["foobar.tar.gz"],
    )


@pytest.fixture
def mock_sample_dependencies(mocker: MockerFixture) -> SampleDependencies:
    return SampleDependencies(
        model_validate_json=mocker.patch(
            "datagen.llm_generator.Document.model_validate_json", autospec=True
        ),
        completion=mocker.patch("datagen.llm_generator.LLM.completion", autospec=True),
        save_training_sample=mocker.patch(
            "datagen.llm_generator.save_training_sample", autospec=True
        ),
    )


@pytest.fixture
def mock_generate_sample_success(
    mocker: MockerFixture, generator: GeneratorFixture
) -> MagicMock:
    return mocker.patch.object(
        generator, "_generate_sample", autospec=True, return_value=True
    )


@pytest.fixture
def mock_generate_sample_mixed(
    mocker: MockerFixture, generator: GeneratorFixture
) -> MagicMock:
    return mocker.patch.object(
        generator, "_generate_sample", autospec=True, side_effect=[True, False, True]
    )


def test_init_downloads_and_extracts_batches(
    mock_generator_dependencies: GeneratorDependencies,
    generator: GeneratorFixture,
) -> None:
    mock_generator_dependencies.download_and_extract.assert_called_once_with(
        filename="foobar.tar.gz", output_folder=Path("./data/documents/foobar")
    )


def test_init_multiple_batches_extends_document_files(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_generator_dependencies: GeneratorDependencies,
) -> None:
    mock_glob: MagicMock = mocker.patch.object(Path, "glob")
    mock_glob.return_value = [
        Path(f"./data/documents/foo/document_{i}.json") for i in range(2)
    ]
    GeneratorFixture(
        system_prompt="foo",
        schema=MagicMock,
        schema_name="baz",
        model_name="qux",
        api_key="quux",
        document_batches=["batch1.tar.gz", "batch2.tar"],
    )
    assert mock_generator_dependencies.download_and_extract.call_count == 2


def test_generate_successful_samples_calls_generate_sample_three_times(
    mock_generate_sample_success: MagicMock, generator: GeneratorFixture
) -> None:
    generator.generate(3)
    assert mock_generate_sample_success.call_count == 3


def test_generate_existing_files_skips_and_generates_remaining(
    mock_filesystem: FileSystem,
    mock_generate_sample_success: MagicMock,
    generator: GeneratorFixture,
) -> None:
    mock_filesystem.exists.side_effect = [True, True, False, False, False]
    generator.generate(5)
    assert mock_generate_sample_success.call_count == 3


def test_generate_sample_failures_continues_generation(
    mock_generate_sample_mixed: MagicMock, generator: GeneratorFixture
) -> None:
    generator.generate(3)
    assert mock_generate_sample_mixed.call_count == 3


def test_generate_exceeds_available_caps_at_document_count(
    mock_generate_sample_success: MagicMock, generator: GeneratorFixture
) -> None:
    generator.generate(10)
    assert mock_generate_sample_success.call_count == 5


def test_generate_sample_valid_input_returns_true(
    mock_sample_dependencies: SampleDependencies, generator: GeneratorFixture
) -> None:
    mock_sample_dependencies.model_validate_json.return_value = MagicMock(
        source="foo",
        content="bar",
    )
    mock_message: MagicMock = MagicMock()
    mock_message.choices = [MagicMock(message=MagicMock(content="baz"))]
    mock_sample_dependencies.completion.return_value = mock_message
    mock_sample_dependencies.save_training_sample.return_value = True
    assert generator.generate_sample(Path("foo.json")) is True
    mock_sample_dependencies.save_training_sample.assert_called_once()


def test_generate_sample_passes_default_sampling_params_to_completion(
    mock_sample_dependencies: SampleDependencies, generator: GeneratorFixture
) -> None:
    mock_sample_dependencies.model_validate_json.return_value = MagicMock(
        source="foo",
        content="bar",
    )
    mock_message: MagicMock = MagicMock()
    mock_message.choices = [MagicMock(message=MagicMock(content="baz"))]
    mock_sample_dependencies.completion.return_value = mock_message
    mock_sample_dependencies.save_training_sample.return_value = True
    generator.generate_sample(Path("foo.json"))
    _, kwargs = mock_sample_dependencies.completion.call_args
    assert kwargs["max_tokens"] == 32768
    assert kwargs["temperature"] == 0.001


def test_generate_sample_passes_custom_sampling_params_to_completion(
    mock_sample_dependencies: SampleDependencies,
    mock_path_operations: PathOperations,
    mock_filesystem: FileSystem,
    mock_generator_dependencies: GeneratorDependencies,
) -> None:
    custom_generator: GeneratorFixture = GeneratorFixture(
        system_prompt="foo",
        schema=MagicMock,
        schema_name="baz",
        model_name="qux",
        api_key="quux",
        document_batches=["foobar.tar.gz"],
        max_tokens=4096,
        temperature=0.7,
    )
    mock_sample_dependencies.model_validate_json.return_value = MagicMock(
        source="foo",
        content="bar",
    )
    mock_message: MagicMock = MagicMock()
    mock_message.choices = [MagicMock(message=MagicMock(content="baz"))]
    mock_sample_dependencies.completion.return_value = mock_message
    mock_sample_dependencies.save_training_sample.return_value = True
    custom_generator.generate_sample(Path("foo.json"))
    _, kwargs = mock_sample_dependencies.completion.call_args
    assert kwargs["max_tokens"] == 4096
    assert kwargs["temperature"] == 0.7


@pytest.mark.parametrize(
    "completion_return",
    [
        None,
        MagicMock(choices=[MagicMock(message=MagicMock(content=None))]),
        MagicMock(choices=[MagicMock(spec=[])]),
    ],
)
def test_generate_sample_failure_conditions_returns_false(
    mock_sample_dependencies: SampleDependencies,
    generator: GeneratorFixture,
    completion_return: MagicMock | None,
) -> None:
    mock_sample_dependencies.model_validate_json.return_value = MagicMock(
        source="foo", content="bar"
    )
    mock_sample_dependencies.completion.return_value = completion_return
    assert generator.generate_sample(Path("foo.json")) is False


def test_generate_sample_save_failure_returns_false(
    mock_sample_dependencies: SampleDependencies, generator: GeneratorFixture
) -> None:
    mock_sample_dependencies.model_validate_json.return_value = MagicMock(
        source="foo",
        content="bar",
    )
    mock_message: MagicMock = MagicMock()
    mock_message.choices = [MagicMock(message=MagicMock(content="baz"))]
    mock_sample_dependencies.completion.return_value = mock_message
    mock_sample_dependencies.save_training_sample.return_value = False
    assert generator.generate_sample(Path("foo.json")) is False


def test_generate_sample_exception_raised_returns_false(
    mock_sample_dependencies: SampleDependencies, generator: GeneratorFixture
) -> None:
    mock_sample_dependencies.model_validate_json.side_effect = Exception("qux")
    assert generator.generate_sample(Path("foo.json")) is False


def test_run_backfill_all_missing_processes_all_documents(
    mock_filesystem: FileSystem,
    mock_generate_sample_success: MagicMock,
    generator: GeneratorFixture,
) -> None:
    generator.run_backfill()
    assert mock_generate_sample_success.call_count == 5


def test_run_backfill_all_exist_skips_all_documents(
    mock_filesystem: FileSystem,
    mock_generate_sample_success: MagicMock,
    generator: GeneratorFixture,
) -> None:
    mock_filesystem.exists.return_value = True
    generator.run_backfill()
    assert mock_generate_sample_success.call_count == 0


def test_run_backfill_mixed_existence_processes_missing_only(
    mock_filesystem: FileSystem,
    mock_generate_sample_success: MagicMock,
    generator: GeneratorFixture,
) -> None:
    mock_filesystem.exists.side_effect = [True, False, True, False, False]
    generator.run_backfill()
    assert mock_generate_sample_success.call_count == 3


def test_run_backfill_generation_failures_continues_processing(
    mock_filesystem: FileSystem,
    mock_generate_sample_mixed: MagicMock,
    generator: GeneratorFixture,
) -> None:
    mock_filesystem.exists.side_effect = [False, False, False]
    generator.run_backfill()
    assert mock_generate_sample_mixed.call_count == 3


def test_run_backfill_document_validation_exception_continues_processing(
    mock_sample_dependencies: SampleDependencies,
    mock_filesystem: FileSystem,
    mock_generate_sample_success: MagicMock,
    generator: GeneratorFixture,
) -> None:
    mock_sample_dependencies.model_validate_json.side_effect = [
        Exception("error"),
        MagicMock(source="foo", content="bar"),
        Exception("error"),
        MagicMock(source="foo", content="bar"),
        MagicMock(source="foo", content="bar"),
    ]
    generator.run_backfill()
    assert mock_generate_sample_success.call_count == 3


def test_run_backfill_empty_document_list_logs_no_missing_samples(
    mocker: MockerFixture,
    mock_filesystem: FileSystem,
    mock_generator_dependencies: GeneratorDependencies,
) -> None:
    empty_generator: GeneratorFixture = GeneratorFixture(
        system_prompt="foo",
        schema=MagicMock,
        schema_name="baz",
        model_name="qux",
        api_key="quux",
        document_batches=[],
    )
    mock_logger: MagicMock = mocker.patch.object(empty_generator.get_logger(), "info")
    empty_generator.run_backfill()
    mock_logger.assert_called_with("No missing samples to backfill")


def test_run_backfill_get_output_filename_exception_increments_failures(
    mock_filesystem: FileSystem,
    mock_generator_dependencies: GeneratorDependencies,
    mock_generate_sample_success: MagicMock,
    generator: GeneratorFixture,
) -> None:
    mock_generator_dependencies.get_output_filename.side_effect = Exception(
        "filename error"
    )
    generator.run_backfill()
    assert mock_generate_sample_success.call_count == 0
