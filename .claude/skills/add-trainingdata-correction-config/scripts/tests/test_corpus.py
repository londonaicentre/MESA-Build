import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, mock_open

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from corpus import Corpus
from utils.llm import BatchOutput


class MockSchema(BaseModel):
    field: str


class CorpusFixture(Corpus):
    @staticmethod
    def load_module(module: str) -> ModuleType:
        return Corpus._import(module)

    @staticmethod
    def read_lines(path: Path) -> Iterator[str]:
        return Corpus._read_lines(path)

    @staticmethod
    def read_text(path: Path) -> str:
        return Corpus._read_text(path)

    @staticmethod
    def write_text(path: Path, text: str) -> None:
        Corpus._write_text(path, text)

    @staticmethod
    def modified(path: Path) -> float:
        return Corpus._modified(path)


@dataclass
class ConstructorMocks:
    getLogger: MagicMock
    basicConfig: MagicMock
    _import: MagicMock


@dataclass
class FileMocks:
    _read_lines: MagicMock
    _read_text: MagicMock
    _write_text: MagicMock
    _modified: MagicMock


@dataclass
class RecordMocks:
    records: MagicMock
    status: MagicMock
    note: MagicMock
    response: MagicMock


def make_record_dict(
    record_id: str = "foo", text: str = "bar", stop_reason: str = "end_turn"
) -> dict[str, object]:
    return {
        "recordId": record_id,
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "corge"}]}
            ],
        },
        "modelOutput": {
            "model": "baz",
            "id": "qux",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


def make_record(
    record_id: str = "foo", text: str = "bar", stop_reason: str = "end_turn"
) -> BatchOutput:
    return BatchOutput.model_validate(make_record_dict(record_id, text, stop_reason))


def make_line(record_id: str = "foo") -> str:
    return json.dumps(make_record_dict(record_id)) + "\n"


@pytest.fixture
def constructor_mocks(mocker: MockerFixture) -> ConstructorMocks:
    return ConstructorMocks(
        getLogger=mocker.patch("corpus.logging.getLogger"),
        basicConfig=mocker.patch("corpus.logging.basicConfig"),
        _import=mocker.patch.object(
            Corpus, "_import", return_value=MagicMock(Schema=MockSchema)
        ),
    )


@pytest.fixture
def file_mocks(mocker: MockerFixture) -> FileMocks:
    return FileMocks(
        _read_lines=mocker.patch.object(Corpus, "_read_lines"),
        _read_text=mocker.patch.object(Corpus, "_read_text"),
        _write_text=mocker.patch.object(Corpus, "_write_text"),
        _modified=mocker.patch.object(Corpus, "_modified"),
    )


@pytest.fixture
def record_mocks(mocker: MockerFixture) -> RecordMocks:
    return RecordMocks(
        records=mocker.patch.object(Corpus, "records"),
        status=mocker.patch.object(Corpus, "status"),
        note=mocker.patch.object(Corpus, "note", return_value="abcdef"),
        response=mocker.patch.object(Corpus, "response", return_value="ghijkl"),
    )


@pytest.fixture
def corpus(constructor_mocks: ConstructorMocks) -> CorpusFixture:
    return CorpusFixture("foo")


class TestFileAccess:
    def test_load_module_name_imports_module(self, mocker: MockerFixture) -> None:
        import_module: MagicMock = mocker.patch("corpus.import_module")
        assert CorpusFixture.load_module("foo") is import_module.return_value
        import_module.assert_called_once_with("foo")

    def test_read_lines_path_yields_lines(self, mocker: MockerFixture) -> None:
        mocker.patch.object(Path, "open", mock_open(read_data="foo\nbar\n"))
        assert list(CorpusFixture.read_lines(Path("baz"))) == ["foo\n", "bar\n"]

    def test_read_text_path_returns_text(self, mocker: MockerFixture) -> None:
        mocker.patch.object(Path, "read_text", return_value="foo")
        assert CorpusFixture.read_text(Path("bar")) == "foo"

    def test_write_text_path_creates_folder_and_writes(
        self, mocker: MockerFixture
    ) -> None:
        mkdir: MagicMock = mocker.patch.object(Path, "mkdir")
        write_text: MagicMock = mocker.patch.object(Path, "write_text")
        CorpusFixture.write_text(Path("foo/bar"), "baz")
        mkdir.assert_called_once_with(parents=True, exist_ok=True)
        write_text.assert_called_once_with("baz")

    def test_modified_path_returns_modification_time(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch.object(Path, "stat", return_value=MagicMock(st_mtime=1.0))
        assert CorpusFixture.modified(Path("foo")) == 1.0


class TestRecordAccess:
    def test_init_schema_package_loads_schema(
        self, corpus: CorpusFixture, constructor_mocks: ConstructorMocks
    ) -> None:
        constructor_mocks._import.assert_called_once_with("foo")
        assert corpus.get_schema() is MockSchema

    @pytest.mark.parametrize(
        "record_ids, expected",
        [(None, ["foo", "bar", "baz"]), ({"bar"}, ["bar"]), ({"ba"}, [])],
    )
    def test_records_record_ids_yields_matching_records(
        self,
        file_mocks: FileMocks,
        record_ids: set[str] | None,
        expected: list[str],
    ) -> None:
        file_mocks._read_lines.return_value = iter(
            [make_line("foo"), make_line("bar"), make_line("baz")]
        )
        assert [record.recordId for _, record in Corpus.records(record_ids)] == expected

    def test_note_record_returns_note_text(self) -> None:
        assert Corpus.note(make_record()) == "corge"

    def test_response_record_returns_response_text(self) -> None:
        assert Corpus.response(make_record(text="foo")) == "foo"

    @pytest.mark.parametrize(
        "validated, data, response, stop_reason, expected",
        [
            (MockSchema(field="foo"), {"field": "foo"}, "foo", "end_turn", "valid"),
            (None, {"field": 1}, "foo", "end_turn", "invalid"),
            (None, None, " ", "end_turn", "empty"),
            (None, None, "foo", "max_tokens", "max_tokens"),
            (None, None, "<output>foo", "end_turn", "has_output"),
            (None, None, "foo", "end_turn", "prose"),
        ],
    )
    def test_status_extraction_returns_status(
        self,
        corpus: CorpusFixture,
        mocker: MockerFixture,
        validated: MockSchema | None,
        data: dict[str, object] | None,
        response: str,
        stop_reason: str,
        expected: str,
    ) -> None:
        mocker.patch("corpus.extract_and_validate_json", return_value=(validated, data))
        mocker.patch.object(Corpus, "response", return_value=response)
        assert corpus.status(make_record(stop_reason=stop_reason)) == expected


class TestCensus:
    def test_census_records_groups_and_writes_record_ids(
        self,
        corpus: CorpusFixture,
        file_mocks: FileMocks,
        record_mocks: RecordMocks,
    ) -> None:
        record_mocks.records.return_value = [
            ("", make_record(record_id)) for record_id in ["foo", "bar", "baz"]
        ]
        record_mocks.status.side_effect = ["valid", "prose", "valid"]
        expected: dict[str, list[str]] = {"valid": ["foo", "baz"], "prose": ["bar"]}
        assert corpus.census() == expected
        file_mocks._write_text.assert_called_once_with(
            Corpus.CENSUS_FILE, json.dumps(expected)
        )


class TestSample:
    def test_sample_stale_census_raises(
        self, corpus: CorpusFixture, file_mocks: FileMocks
    ) -> None:
        file_mocks._modified.side_effect = [2.0, 1.0]
        with pytest.raises(RuntimeError):
            corpus.sample("prose", 10, 2)

    @pytest.mark.parametrize("size, expected_sizes", [(10, [2, 1]), (1, [1])])
    def test_sample_status_writes_manifests(
        self,
        corpus: CorpusFixture,
        file_mocks: FileMocks,
        size: int,
        expected_sizes: list[int],
    ) -> None:
        file_mocks._modified.side_effect = [1.0, 2.0]
        file_mocks._read_text.return_value = json.dumps(
            {"prose": ["foo", "bar", "baz"]}
        )
        manifests: list[Path] = corpus.sample("prose", size, 2)
        assert manifests == [
            Corpus.SCRATCH_FOLDER / f"prose_{index + 1}.json"
            for index in range(len(expected_sizes))
        ]
        written: list[list[str]] = [
            json.loads(call.args[1]) for call in file_mocks._write_text.call_args_list
        ]
        assert [len(record_ids) for record_ids in written] == expected_sizes
        assert {record_id for record_ids in written for record_id in record_ids} <= {
            "foo",
            "bar",
            "baz",
        }


class TestShow:
    def test_show_record_ids_renders_truncated_records(
        self, corpus: CorpusFixture, record_mocks: RecordMocks
    ) -> None:
        record_mocks.records.return_value = [("", make_record("foo"))]
        record_mocks.status.return_value = "prose"
        assert corpus.show(["foo"], 3) == "=== foo (prose)\nNOTE\nabc\nRESPONSE\nghi"
        record_mocks.records.assert_called_once_with({"foo"})


class TestTestPrompt:
    def test_test_prompt_record_ids_corrects_copies_and_reports(
        self,
        corpus: CorpusFixture,
        mocker: MockerFixture,
        constructor_mocks: ConstructorMocks,
        file_mocks: FileMocks,
        record_mocks: RecordMocks,
    ) -> None:
        batch_output_corrector: MagicMock = mocker.patch("corpus.BatchOutputCorrector")
        record_mocks.records.return_value = [("bar\n", make_record("foo"))]
        record_mocks.status.side_effect = ["prose", "valid"]
        file_mocks._read_text.return_value = make_line("foo")
        test_file: Path = Corpus.SCRATCH_FOLDER / "prompt_test.jsonl.out"
        corpus.test_prompt(Path("baz.txt"), ["foo"], 3)
        file_mocks._write_text.assert_called_once_with(test_file, "bar\n")
        batch_output_corrector.assert_called_once_with(MockSchema, test_file)
        batch_output_corrector.return_value.correct_with_claude.assert_called_once_with(
            Path("baz.txt"),
            constructor_mocks._import.return_value.PromptBuilder.return_value.build_datagen_prompt.return_value,
        )
        constructor_mocks.getLogger.return_value.info.assert_called_once_with(
            "=== foo: prose -> valid\nghi"
        )


class TestMain:
    @pytest.mark.parametrize(
        "arguments, method, expected",
        [
            (["census"], "census", ()),
            (
                ["sample", "prose", "--size", "5", "--batch", "2"],
                "sample",
                ("prose", 5, 2),
            ),
            (["show", "foo", "bar", "--chars", "10"], "show", (["foo", "bar"], 10)),
            (
                ["test-prompt", "baz.txt", "foo"],
                "test_prompt",
                (Path("baz.txt"), ["foo"], 1500),
            ),
        ],
    )
    def test_main_command_runs_method(
        self,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        constructor_mocks: ConstructorMocks,
        arguments: list[str],
        method: str,
        expected: tuple[object, ...],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["corpus.py", "--schema", "foo", *arguments])
        called: MagicMock = mocker.patch.object(Corpus, method)
        Corpus.main()
        called.assert_called_once_with(*expected)
