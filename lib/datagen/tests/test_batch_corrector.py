import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from datagen.batch_corrector import BatchOutputCorrector


class MockSchema(BaseModel):
    field: str


class BatchOutputCorrectorFixture(BatchOutputCorrector):
    def read(self) -> list[dict[str, object]]:
        return self._read()

    def write(self, records: list[dict[str, object]]) -> None:
        self._write(records)

    @staticmethod
    def get_note(record: dict[str, object]) -> str:
        return BatchOutputCorrector._get_note(record)

    @staticmethod
    def get_response(record: dict[str, object]) -> str:
        return BatchOutputCorrector._get_response(record)

    @staticmethod
    def replace_response(record: dict[str, object], response: str) -> dict[str, object]:
        return BatchOutputCorrector._replace_response(record, response)

    @staticmethod
    def run_claude(system_prompt: str, user_prompt: str) -> str:
        return BatchOutputCorrector._run_claude(system_prompt, user_prompt)


@dataclass
class FileMocks:
    read_text: MagicMock
    write_text: MagicMock
    replace: MagicMock


@dataclass
class CorrectionMocks:
    _read: MagicMock
    _write: MagicMock
    _get_note: MagicMock
    _get_response: MagicMock
    _replace_response: MagicMock
    extract_and_validate_json: MagicMock


@dataclass
class ClaudeMocks:
    read_text: MagicMock
    _run_claude: MagicMock


def make_record(text: str | None = "foo") -> dict[str, object]:
    return {
        "recordId": "quux",
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "system": "bar",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "corge"}]}
            ],
        },
        "modelOutput": {
            "model": "baz",
            "id": "qux",
            "type": "message",
            "role": "assistant",
            "content": [] if text is None else [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


@pytest.fixture
def file_mocks(mocker: MockerFixture) -> FileMocks:
    return FileMocks(
        read_text=mocker.patch.object(
            Path, "read_text", return_value='{"foo": 1}\n{"bar": 2}\n'
        ),
        write_text=mocker.patch.object(Path, "write_text"),
        replace=mocker.patch("datagen.batch_corrector.os.replace", autospec=True),
    )


@pytest.fixture
def correction_mocks(mocker: MockerFixture) -> CorrectionMocks:
    return CorrectionMocks(
        _read=mocker.patch.object(
            BatchOutputCorrector,
            "_read",
            autospec=True,
            return_value=[
                {"recordId": "foo"},
                {"recordId": "bar"},
                {"recordId": "baz"},
            ],
        ),
        _write=mocker.patch.object(BatchOutputCorrector, "_write", autospec=True),
        _get_note=mocker.patch.object(
            BatchOutputCorrector, "_get_note", return_value="qux"
        ),
        _get_response=mocker.patch.object(
            BatchOutputCorrector, "_get_response", return_value="quux"
        ),
        _replace_response=mocker.patch.object(
            BatchOutputCorrector,
            "_replace_response",
            side_effect=lambda record, response: {**record, "text": response},
        ),
        extract_and_validate_json=mocker.patch(
            "datagen.batch_corrector.extract_and_validate_json", autospec=True
        ),
    )


@pytest.fixture
def claude_mocks(mocker: MockerFixture) -> ClaudeMocks:
    return ClaudeMocks(
        read_text=mocker.patch.object(
            Path, "read_text", return_value="corge {EXTRACTION_PROMPT}"
        ),
        _run_claude=mocker.patch.object(
            BatchOutputCorrector, "_run_claude", return_value="grault"
        ),
    )


@pytest.fixture
def corrector(mocker: MockerFixture) -> BatchOutputCorrectorFixture:
    mocker.patch("datagen.batch_corrector.logging.getLogger")
    return BatchOutputCorrectorFixture(MockSchema, Path("foo.jsonl.out"))


class TestFileAccess:
    def test_read_output_file_returns_records(
        self, corrector: BatchOutputCorrectorFixture, file_mocks: FileMocks
    ) -> None:
        assert corrector.read() == [{"foo": 1}, {"bar": 2}]

    def test_write_records_replaces_output_file(
        self, corrector: BatchOutputCorrectorFixture, file_mocks: FileMocks
    ) -> None:
        corrector.write([{"foo": 1}, {"bar": 2}])
        file_mocks.write_text.assert_called_once_with('{"foo": 1}\n{"bar": 2}\n')
        file_mocks.replace.assert_called_once_with(
            Path("foo.jsonl.tmp"), Path("foo.jsonl.out")
        )


class TestRecordAccess:
    def test_get_note_record_returns_note_text(self) -> None:
        assert BatchOutputCorrectorFixture.get_note(make_record()) == "corge"

    @pytest.mark.parametrize("text, expected", [("foo", "foo"), (None, "")])
    def test_get_response_record_returns_response_text(
        self, text: str | None, expected: str
    ) -> None:
        assert BatchOutputCorrectorFixture.get_response(make_record(text)) == expected

    def test_replace_response_record_replaces_content(self) -> None:
        record: dict[str, object] = make_record()
        updated: dict[str, object] = BatchOutputCorrectorFixture.replace_response(
            record, "bar"
        )
        assert BatchOutputCorrectorFixture.get_response(updated) == "bar"
        assert BatchOutputCorrectorFixture.get_response(record) == "foo"

    def test_replace_response_no_model_output_raises(self) -> None:
        with pytest.raises(ValueError):
            BatchOutputCorrectorFixture.replace_response({"modelOutput": None}, "bar")


class TestRunClaude:
    def test_run_claude_prompts_returns_stripped_output(
        self, mocker: MockerFixture
    ) -> None:
        mock_run: MagicMock = mocker.patch(
            "datagen.batch_corrector.subprocess.run", autospec=True
        )
        mock_run.return_value.stdout = " <output>foo</output>\n"
        assert (
            BatchOutputCorrectorFixture.run_claude("foo", "bar")
            == "<output>foo</output>"
        )
        arguments: list[str] = mock_run.call_args.args[0]
        assert arguments[:4] == ["claude", "-p", "--model", "haiku"]
        assert arguments[arguments.index("--system-prompt") + 1] == "foo"
        assert mock_run.call_args.kwargs["input"] == "bar"
        assert mock_run.call_args.kwargs["check"]
        assert mock_run.call_args.kwargs["env"]["MAX_THINKING_TOKENS"] == "3000"


class TestCorrectWithClaude:
    def test_correct_with_claude_invalid_records_corrects_only_those(
        self,
        corrector: BatchOutputCorrectorFixture,
        correction_mocks: CorrectionMocks,
        claude_mocks: ClaudeMocks,
    ) -> None:
        correction_mocks.extract_and_validate_json.side_effect = [
            (None, None),
            (MockSchema(field="foo"), {"field": "foo"}),
            (None, {"foo": 1}),
        ]
        assert corrector.correct_with_claude(Path("foo.txt"), "bar") == 2
        assert claude_mocks._run_claude.call_args_list[0].args == (
            "corge bar",
            "<note>\nqux\n</note>\n\n<original_response>\nquux\n</original_response>",
        )
        assert correction_mocks._write.call_count == 2
        correction_mocks._write.assert_called_with(
            corrector,
            [
                {"recordId": "foo", "text": "grault"},
                {"recordId": "bar"},
                {"recordId": "baz", "text": "grault"},
            ],
        )

    def test_correct_with_claude_many_invalid_records_writes_each_correction(
        self,
        corrector: BatchOutputCorrectorFixture,
        correction_mocks: CorrectionMocks,
        claude_mocks: ClaudeMocks,
    ) -> None:
        correction_mocks._read.return_value = [
            {"recordId": index} for index in range(12)
        ]
        correction_mocks.extract_and_validate_json.return_value = (None, None)
        assert corrector.correct_with_claude(Path("foo.txt"), "bar") == 12
        assert correction_mocks._write.call_count == 12

    def test_correct_with_claude_no_invalid_records_does_nothing(
        self,
        corrector: BatchOutputCorrectorFixture,
        correction_mocks: CorrectionMocks,
        claude_mocks: ClaudeMocks,
    ) -> None:
        correction_mocks.extract_and_validate_json.return_value = (
            MockSchema(field="foo"),
            {"field": "foo"},
        )
        assert corrector.correct_with_claude(Path("foo.txt"), "bar") == 0
        claude_mocks._run_claude.assert_not_called()
        correction_mocks._write.assert_not_called()


class TestCorrectWithRules:
    @pytest.fixture
    def mock_rules(self, mocker: MockerFixture) -> MagicMock:
        mock_rules: MagicMock = MagicMock()
        mocker.patch(
            "datagen.batch_corrector.CorrectionRules.load", return_value=mock_rules
        )
        return mock_rules

    def test_correct_with_rules_valid_records_rewrites_corrected_outputs(
        self,
        corrector: BatchOutputCorrectorFixture,
        correction_mocks: CorrectionMocks,
        mock_rules: MagicMock,
    ) -> None:
        def correct(
            sample: dict[str, object], note: str
        ) -> tuple[dict[str, object], bool]:
            if sample["field"] != "foo":
                return sample, False
            return {**sample, "field": "bar"}, True

        mock_rules.apply.side_effect = correct
        correction_mocks.extract_and_validate_json.side_effect = [
            (MockSchema(field="foo"), None),
            (None, None),
            (MockSchema(field="baz"), None),
        ]
        assert corrector.correct_with_rules(Path("foo.yaml")) == 1
        correction_mocks._write.assert_called_once_with(
            corrector,
            [
                {
                    "recordId": "foo",
                    "text": "<output>\n"
                    + json.dumps({"field": "bar"}, indent=2)
                    + "\n</output>",
                },
                {"recordId": "bar"},
                {"recordId": "baz"},
            ],
        )
        assert mock_rules.apply.call_args_list[0].args[1] == "qux"

    def test_correct_with_rules_invalid_correction_raises(
        self,
        corrector: BatchOutputCorrectorFixture,
        correction_mocks: CorrectionMocks,
        mock_rules: MagicMock,
    ) -> None:
        def correct(
            sample: dict[str, object], note: str
        ) -> tuple[dict[str, object], bool]:
            return {**sample, "field": None}, True

        mock_rules.apply.side_effect = correct
        correction_mocks.extract_and_validate_json.return_value = (
            MockSchema(field="foo"),
            None,
        )
        with pytest.raises(ValidationError):
            corrector.correct_with_rules(Path("foo.yaml"))
        correction_mocks._write.assert_not_called()
