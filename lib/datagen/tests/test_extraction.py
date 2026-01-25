from dataclasses import dataclass
from unittest.mock import MagicMock, mock_open

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from datagen.extraction import (
    get_output_filename,
    _try_parse_and_validate,
    _save_json_file,
    extract_and_validate_json,
    save_training_sample,
)
from mesa_types import Document


class MockSchema(BaseModel):
    field: str


@dataclass
class FileSystemMocks:
    makedirs: MagicMock
    open: MagicMock
    json_dump: MagicMock


@dataclass
class ParseValidationMocks:
    try_parse_and_validate: MagicMock


@dataclass
class ExtractionMocks:
    extract_and_validate_json: MagicMock
    get_output_filename: MagicMock
    save_json_file: MagicMock


@pytest.fixture
def mock_filesystem(mocker: MockerFixture) -> FileSystemMocks:
    mock_file = mock_open()
    return FileSystemMocks(
        makedirs=mocker.patch("os.makedirs", autospec=True),
        open=mocker.patch("builtins.open", mock_file),
        json_dump=mocker.patch("json.dump", autospec=True),
    )


@pytest.fixture
def mock_extraction_dependencies(mocker: MockerFixture) -> ExtractionMocks:
    return ExtractionMocks(
        extract_and_validate_json=mocker.patch(
            "datagen.extraction.extract_and_validate_json", autospec=True
        ),
        get_output_filename=mocker.patch(
            "datagen.extraction.get_output_filename",
            autospec=True,
            return_value="foo_bar_abc12345.json",
        ),
        save_json_file=mocker.patch(
            "datagen.extraction._save_json_file", autospec=True, return_value=True
        ),
    )


@pytest.fixture
def mock_parse_validation(mocker: MockerFixture) -> ParseValidationMocks:
    return ParseValidationMocks(
        try_parse_and_validate=mocker.patch(
            "datagen.extraction._try_parse_and_validate", autospec=True
        ),
    )


class TestGetOutputFilename:
    def test_get_output_filename_valid_input_returns_formatted_string(self) -> None:
        assert (
            get_output_filename(
                "foo_schema", "v1", Document(content="foo", source="bar", timestamp="")
            )
            == "foo_schemav1_bar_acbd18db.json"
        )

    def test_get_output_filename_same_src_returns_same_hash(self) -> None:
        assert (
            get_output_filename(
                "foo_schema", "v1", Document(content="foo", source="bar", timestamp="")
            ).split("_")[-1]
            == get_output_filename(
                "foo_schema", "v1", Document(content="foo", source="baz", timestamp="")
            ).split("_")[-1]
        )

    def test_get_output_filename_different_src_returns_different_hash(self) -> None:
        assert get_output_filename(
            "foo_schema", "v1", Document(content="foo", source="bar", timestamp="")
        ) != get_output_filename(
            "foo_schema", "v1", Document(content="baz", source="bar", timestamp="")
        )


class TestTryParseAndValidate:
    def test_try_parse_and_validate_valid_json_valid_schema_returns_both(self) -> None:
        validated, data = _try_parse_and_validate('{"field": "foo"}', MockSchema)
        assert validated is not None
        assert validated.field == "foo"
        assert data == {"field": "foo"}

    def test_try_parse_and_validate_valid_json_invalid_schema_returns_none_and_data(
        self,
    ) -> None:
        validated, data = _try_parse_and_validate('{"qux": "foo"}', MockSchema)
        assert validated is None
        assert data == {"qux": "foo"}

    def test_try_parse_and_validate_invalid_json_returns_none_both(self) -> None:
        validated, data = _try_parse_and_validate("foobar", MockSchema)
        assert validated is None
        assert data is None

    def test_try_parse_and_validate_json_array_returns_none_both(self) -> None:
        validated, data = _try_parse_and_validate('[{"field": "foo"}]', MockSchema)
        assert validated is None
        assert data is None


class TestSaveJsonFile:
    def test_save_json_file_success_returns_true(
        self, mock_filesystem: FileSystemMocks
    ) -> None:
        assert _save_json_file({"foo": "bar"}, "/path/file.json") is True
        mock_filesystem.json_dump.assert_called_once()

    def test_save_json_file_exception_returns_false(
        self, mock_filesystem: FileSystemMocks
    ) -> None:
        mock_filesystem.open.side_effect = IOError("error")
        assert _save_json_file({"foo": "bar"}, "/path/file.json") is False


class TestExtractAndValidateJson:
    def test_extract_and_validate_json_output_tags_valid_returns_validated(
        self, mock_parse_validation: ParseValidationMocks
    ) -> None:
        mock_validated = MagicMock()
        mock_validated.field = "foo"
        mock_parse_validation.try_parse_and_validate.return_value = (
            mock_validated,
            {"field": "foo"},
        )
        validated, data = extract_and_validate_json(
            '<output>{"field": "foo"}</output>', MockSchema
        )
        assert validated is mock_validated
        assert data == {"field": "foo"}

    def test_extract_and_validate_json_output_tags_invalid_schema_returns_data_only(
        self, mock_parse_validation: ParseValidationMocks
    ) -> None:
        mock_parse_validation.try_parse_and_validate.return_value = (
            None,
            {"qux": "foo"},
        )
        validated, data = extract_and_validate_json(
            '<output>{"qux": "foo"}</output>', MockSchema
        )
        assert validated is None
        assert data == {"qux": "foo"}

    def test_extract_and_validate_json_output_tags_invalid_json_falls_back(
        self, mock_parse_validation: ParseValidationMocks
    ) -> None:
        mock_validated = MagicMock()
        mock_validated.field = "bar"
        mock_parse_validation.try_parse_and_validate.side_effect = [
            (None, None),
            (None, None),
            (mock_validated, {"field": "bar"}),
        ]
        validated, _ = extract_and_validate_json(
            '<output>foobar</output>\n{"field": "bar"}', MockSchema
        )
        assert validated is mock_validated

    def test_extract_and_validate_json_whole_response_valid_returns_validated(
        self, mock_parse_validation: ParseValidationMocks
    ) -> None:
        mock_validated = MagicMock()
        mock_validated.field = "foo"
        mock_parse_validation.try_parse_and_validate.return_value = (
            mock_validated,
            {"field": "foo"},
        )
        validated, data = extract_and_validate_json('{"field": "foo"}', MockSchema)
        assert validated is mock_validated

    def test_extract_and_validate_json_whole_response_invalid_schema_returns_data_only(
        self, mock_parse_validation: ParseValidationMocks
    ) -> None:
        mock_parse_validation.try_parse_and_validate.return_value = (
            None,
            {"qux": "foo"},
        )
        validated, data = extract_and_validate_json('{"qux": "foo"}', MockSchema)
        assert validated is None
        assert data == {"qux": "foo"}

    def test_extract_and_validate_json_regex_fallback_valid_returns_validated(
        self, mock_parse_validation: ParseValidationMocks
    ) -> None:
        mock_validated = MagicMock()
        mock_validated.field = "foo"
        mock_parse_validation.try_parse_and_validate.side_effect = [
            (None, None),
            (mock_validated, {"field": "foo"}),
        ]
        validated, _ = extract_and_validate_json(
            'corge {"field": "foo"} grault', MockSchema
        )
        assert validated is mock_validated

    def test_extract_and_validate_json_regex_fallback_invalid_schema_returns_data_only(
        self, mock_parse_validation: ParseValidationMocks
    ) -> None:
        mock_parse_validation.try_parse_and_validate.side_effect = [
            (None, None),
            (None, {"qux": "foo"}),
        ]
        validated, data = extract_and_validate_json(
            'corge {"qux": "foo"} grault', MockSchema
        )
        assert validated is None
        assert data == {"qux": "foo"}

    def test_extract_and_validate_json_no_json_returns_none_both(
        self, mock_parse_validation: ParseValidationMocks, mocker: MockerFixture
    ) -> None:
        mocker.patch("datagen.extraction.logger")
        mock_parse_validation.try_parse_and_validate.return_value = (None, None)
        validated, data = extract_and_validate_json("foobar", MockSchema)
        assert validated is None
        assert data is None


class TestSaveTrainingSample:
    def test_save_training_sample_valid_output_saves_and_returns_true(
        self, mock_extraction_dependencies: ExtractionMocks, mocker: MockerFixture
    ) -> None:
        mocker.patch("os.makedirs", autospec=True)
        mocker.patch("builtins.print")
        mock_validated = MagicMock()
        mock_validated.model_dump.return_value = {"field": "foo"}
        mock_extraction_dependencies.extract_and_validate_json.return_value = (
            mock_validated,
            {"field": "foo"},
        )
        assert (
            save_training_sample(
                '{"field": "foo"}',
                "bar",
                "foobarbaz",
                MockSchema,
                "foo_schema",
                "v1",
                "/output",
            )
            is True
        )
        mock_extraction_dependencies.save_json_file.assert_called_once()

    def test_save_training_sample_invalid_schema_saves_to_invalid_and_returns_false(
        self, mock_extraction_dependencies: ExtractionMocks, mocker: MockerFixture
    ) -> None:
        mocker.patch("os.makedirs", autospec=True)
        mocker.patch("builtins.print")
        mocker.patch("datagen.extraction.logger")
        mock_extraction_dependencies.extract_and_validate_json.return_value = (
            None,
            {"qux": "foo"},
        )
        assert (
            save_training_sample(
                '{"qux": "foo"}',
                "bar",
                "foobarbaz",
                MockSchema,
                "foo_schema",
                "v1",
                "/output",
            )
            is False
        )
        mock_extraction_dependencies.save_json_file.assert_called_once()

    def test_save_training_sample_no_json_returns_false(
        self, mock_extraction_dependencies: ExtractionMocks, mocker: MockerFixture
    ) -> None:
        mocker.patch("os.makedirs", autospec=True)
        mocker.patch("builtins.print")
        mocker.patch("datagen.extraction.logger")
        mock_extraction_dependencies.extract_and_validate_json.return_value = (
            None,
            None,
        )
        assert (
            save_training_sample(
                '{"field": "foo"}',
                "bar",
                "foobarbaz",
                MockSchema,
                "foo_schema",
                "v1",
                "/output",
            )
            is False
        )
        mock_extraction_dependencies.save_json_file.assert_not_called()

    def test_save_training_sample_save_failure_returns_false(
        self, mock_extraction_dependencies: ExtractionMocks, mocker: MockerFixture
    ) -> None:
        mocker.patch("os.makedirs", autospec=True)
        mocker.patch("builtins.print")
        mock_validated = MagicMock()
        mock_validated.model_dump.return_value = {"field": "foo"}
        mock_extraction_dependencies.extract_and_validate_json.return_value = (
            mock_validated,
            {"field": "foo"},
        )
        mock_extraction_dependencies.save_json_file.return_value = False
        assert (
            save_training_sample(
                '{"field": "foo"}',
                "bar",
                "foobarbaz",
                MockSchema,
                "foo_schema",
                "v1",
                "/output",
            )
            is False
        )

    def test_save_training_sample_creates_output_directories(
        self, mock_extraction_dependencies: ExtractionMocks, mocker: MockerFixture
    ) -> None:
        mock_makedirs = mocker.patch("os.makedirs", autospec=True)
        mocker.patch("builtins.print")
        mock_extraction_dependencies.extract_and_validate_json.return_value = (
            None,
            None,
        )
        mocker.patch("datagen.extraction.logger")
        save_training_sample(
            '{"field": "foo"}',
            "bar",
            "foobarbaz",
            MockSchema,
            "foo_schema",
            "v1",
            "/output",
        )
        assert mock_makedirs.call_count == 2
