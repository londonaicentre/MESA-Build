import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docsynth.types.correction_rules import (
    DocumentCorrectionAction,
    DocumentCorrectionRule,
)
from docsynth.types.documents import DocsynthDocument
from pytest_mock import MockerFixture

from documents import Documents


class DocumentsFixture(Documents):
    @staticmethod
    def read_text(path: Path) -> str:
        return Documents._read_text(path)

    @staticmethod
    def write_text(path: Path, text: str) -> None:
        Documents._write_text(path, text)

    @staticmethod
    def reset_folder(folder: Path) -> None:
        Documents._reset_folder(folder)

    def doc_ids(self) -> list[str]:
        return self._doc_ids()

    @staticmethod
    def read_document(folder: Path, doc_id: str) -> DocsynthDocument:
        return Documents._read_document(folder, doc_id)


@dataclass
class ConstructorMocks:
    getLogger: MagicMock
    basicConfig: MagicMock
    _read_text: MagicMock


@dataclass
class DocumentMocks:
    _read_document: MagicMock
    _write_text: MagicMock


@dataclass
class PromptTestMocks:
    _reset_folder: MagicMock
    Corrector: MagicMock
    load: MagicMock


def make_document(
    doc_id: str = "foo", content: str = "bar", prompt: str = "baz"
) -> DocsynthDocument:
    return DocsynthDocument(
        doc_id=doc_id,
        document_name="qux",
        profile="quux",
        prompt=prompt,
        content=content,
        source="corge",
        timestamp="20250101_000000_000",
    )


@pytest.fixture
def constructor_mocks(mocker: MockerFixture) -> ConstructorMocks:
    return ConstructorMocks(
        getLogger=mocker.patch("documents.logging.getLogger"),
        basicConfig=mocker.patch("documents.logging.basicConfig"),
        _read_text=mocker.patch.object(
            Documents, "_read_text", return_value="output:\n  subdirectory: foo\n"
        ),
    )


@pytest.fixture
def document_mocks(mocker: MockerFixture) -> DocumentMocks:
    return DocumentMocks(
        _read_document=mocker.patch.object(Documents, "_read_document"),
        _write_text=mocker.patch.object(Documents, "_write_text"),
    )


@pytest.fixture
def test_prompt_mocks(mocker: MockerFixture) -> PromptTestMocks:
    return PromptTestMocks(
        _reset_folder=mocker.patch.object(Documents, "_reset_folder"),
        Corrector=mocker.patch("documents.Corrector"),
        load=mocker.patch("documents.DocumentCorrectionRules.load"),
    )


@pytest.fixture
def documents(constructor_mocks: ConstructorMocks) -> DocumentsFixture:
    return DocumentsFixture()


class TestFileAccess:
    def test_read_text_path_returns_text(self, tmp_path: Path) -> None:
        (tmp_path / "foo").write_text("bar")
        assert DocumentsFixture.read_text(tmp_path / "foo") == "bar"

    def test_write_text_path_creates_folder_and_writes(self, tmp_path: Path) -> None:
        DocumentsFixture.write_text(tmp_path / "foo" / "bar", "baz")
        assert (tmp_path / "foo" / "bar").read_text() == "baz"

    @pytest.mark.parametrize("exists", [False, True])
    def test_reset_folder_folder_leaves_empty_folder(
        self, tmp_path: Path, exists: bool
    ) -> None:
        if exists:
            (tmp_path / "foo").mkdir()
            (tmp_path / "foo" / "bar").write_text("baz")
        DocumentsFixture.reset_folder(tmp_path / "foo")
        assert list((tmp_path / "foo").iterdir()) == []


class TestDocumentAccess:
    def test_init_pipeline_reads_pipeline_config(
        self, documents: DocumentsFixture, constructor_mocks: ConstructorMocks
    ) -> None:
        constructor_mocks._read_text.assert_called_once_with(Path("pipeline.yml"))

    def test_doc_ids_folder_returns_sorted_ids_without_metadata(
        self, documents: DocumentsFixture, mocker: MockerFixture
    ) -> None:
        glob: MagicMock = mocker.patch.object(
            Path,
            "glob",
            return_value=[Path("bar.json"), Path("metadata.json"), Path("baz.json")],
        )
        assert documents.doc_ids() == ["bar", "baz"]
        glob.assert_called_once_with("*.json")

    def test_read_document_doc_id_parses_document(
        self, constructor_mocks: ConstructorMocks
    ) -> None:
        constructor_mocks._read_text.return_value = make_document().model_dump_json()
        assert DocumentsFixture.read_document(Path("bar"), "foo") == make_document()
        constructor_mocks._read_text.assert_called_once_with(Path("bar/foo.json"))

    @pytest.mark.parametrize(
        "structure", ["", "\n## MIMIC THIS DOCUMENT STRUCTURE\n\ngrault\n"]
    )
    def test_instructions_prompt_returns_sections_without_structure(
        self, structure: str
    ) -> None:
        assert (
            Documents.instructions(
                make_document(
                    prompt="foo\n\n# SPECIFIC INSTRUCTIONS\n\n## BAR\n\nbaz\n"
                    f"\n## QUX\n\nquux\n{structure}\n# YOUR TASK\n\ncorge"
                )
            )
            == "## BAR\n\nbaz\n\n## QUX\n\nquux"
        )


class TestSample:
    @pytest.mark.parametrize("size, expected_sizes", [(10, [2, 1]), (1, [1])])
    def test_sample_size_writes_manifests(
        self,
        documents: DocumentsFixture,
        mocker: MockerFixture,
        document_mocks: DocumentMocks,
        size: int,
        expected_sizes: list[int],
    ) -> None:
        mocker.patch.object(Documents, "_doc_ids", return_value=["foo", "bar", "baz"])
        assert documents.sample(size, 2) == [
            Documents.SCRATCH_FOLDER / f"documents_{index + 1}.json"
            for index in range(len(expected_sizes))
        ]
        written: list[list[str]] = [
            json.loads(call.args[1])
            for call in document_mocks._write_text.call_args_list
        ]
        assert [len(doc_ids) for doc_ids in written] == expected_sizes
        assert {doc_id for doc_ids in written for doc_id in doc_ids} <= {
            "foo",
            "bar",
            "baz",
        }


class TestShow:
    def test_show_doc_ids_renders_instructions_and_truncated_content(
        self,
        documents: DocumentsFixture,
        mocker: MockerFixture,
        document_mocks: DocumentMocks,
    ) -> None:
        mocker.patch.object(Documents, "instructions", return_value="grault")
        document_mocks._read_document.return_value = make_document(content="abcdef")
        assert (
            documents.show(["foo"], 3) == "=== foo\nINSTRUCTIONS\ngrault\nCONTENT\nabc"
        )
        document_mocks._read_document.assert_called_once_with(Path("output/foo"), "foo")


class TestTestPrompt:
    def test_test_prompt_doc_ids_corrects_copies_and_reports(
        self,
        documents: DocumentsFixture,
        constructor_mocks: ConstructorMocks,
        document_mocks: DocumentMocks,
        test_prompt_mocks: PromptTestMocks,
    ) -> None:
        rules_file: Path = Documents.TEST_FOLDER / "document_correction.yaml"
        document_mocks._read_document.side_effect = [
            make_document(content="bar\nbaz"),
            make_document(content="bar\nqux"),
        ]
        test_prompt_mocks.load.return_value.rules = [
            DocumentCorrectionRule(
                name="foo-1",
                description="grault",
                document="foo",
                action=DocumentCorrectionAction.REPLACE,
                find="baz",
                value="qux",
            ),
            DocumentCorrectionRule(
                name="garply-1",
                description="waldo",
                document="garply",
                action=DocumentCorrectionAction.REGENERATE,
            ),
        ]
        documents.test_prompt(Path("quux.txt"), ["foo"], 15)
        test_prompt_mocks._reset_folder.assert_called_once_with(Documents.TEST_FOLDER)
        document_mocks._write_text.assert_called_once_with(
            Documents.TEST_FOLDER / "foo.json",
            make_document(content="bar\nbaz").model_dump_json(indent=2),
        )
        test_prompt_mocks.Corrector.assert_called_once_with(
            Documents.TEST_FOLDER, rules_file
        )
        corrector: MagicMock = test_prompt_mocks.Corrector.return_value
        corrector.review.assert_called_once_with(Path("quux.txt"))
        corrector.correct_with_rules.assert_called_once_with()
        corrector.correct_with_claude.assert_called_once_with()
        test_prompt_mocks.load.assert_called_once_with(rules_file)
        constructor_mocks.getLogger.return_value.info.assert_called_once_with(
            "=== foo\nreplace: grault\n@@ -1,2 +1,2 @@"
        )


class TestMain:
    @pytest.mark.parametrize(
        "arguments, method, expected",
        [
            (["sample", "--size", "5", "--batch", "2"], "sample", (5, 2)),
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
        monkeypatch.setattr(sys, "argv", ["documents.py", *arguments])
        called: MagicMock = mocker.patch.object(Documents, method)
        Documents.main()
        called.assert_called_once_with(*expected)
