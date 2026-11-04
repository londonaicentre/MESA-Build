import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docsynth.types.documents import DocsynthDocument
from pytest_mock import MockerFixture
from yaml import safe_load

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

    @staticmethod
    def supplement(folder: Path) -> int:
        return Documents._supplement(folder)

    @staticmethod
    def doc_ids(folder: Path) -> list[str]:
        return Documents._doc_ids(folder)

    @staticmethod
    def render(folder: Path, doc_ids: list[str], chars: int) -> str:
        return Documents._render(folder, doc_ids, chars)

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
class SupplementTestMocks:
    _reset_folder: MagicMock
    _supplement: MagicMock
    _doc_ids: MagicMock
    _render: MagicMock


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
def supplement_test_mocks(mocker: MockerFixture) -> SupplementTestMocks:
    return SupplementTestMocks(
        _reset_folder=mocker.patch.object(Documents, "_reset_folder"),
        _supplement=mocker.patch.object(Documents, "_supplement"),
        _doc_ids=mocker.patch.object(Documents, "_doc_ids", return_value=["foo"]),
        _render=mocker.patch.object(Documents, "_render", return_value="bar"),
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

    def test_supplement_empty_supplement_generates_nothing(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "pipeline.yml").write_text(
            "llm:\n  enabled: false\n  provider: none\n"
            "profile_selection:\n  domain: chestrad\n  mode: random\n"
            "  count: 1\n  file: null\n"
            "structure_selection:\n  enabled_structures: null\n"
            "prompt_config:\n  include_style: true\n  include_content: true\n"
            "output:\n  subdirectory: foo\n"
        )
        (tmp_path / "supplement.yaml").write_text("supplements: []\n")
        assert DocumentsFixture.supplement(tmp_path) == 0


class TestDocumentAccess:
    def test_init_pipeline_reads_pipeline_config(
        self, documents: DocumentsFixture, constructor_mocks: ConstructorMocks
    ) -> None:
        constructor_mocks._read_text.assert_called_once_with(Path("pipeline.yml"))

    def test_doc_ids_folder_returns_sorted_ids_without_metadata(
        self, tmp_path: Path
    ) -> None:
        for name in ["bar.json", "metadata.json", "baz.json", "qux.txt"]:
            (tmp_path / name).write_text("")
        assert DocumentsFixture.doc_ids(tmp_path) == ["bar", "baz"]

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
    def test_render_doc_ids_renders_instructions_and_truncated_content(
        self, mocker: MockerFixture, document_mocks: DocumentMocks
    ) -> None:
        mocker.patch.object(Documents, "instructions", return_value="grault")
        document_mocks._read_document.return_value = make_document(content="abcdef")
        assert (
            DocumentsFixture.render(Path("bar"), ["foo"], 3)
            == "=== foo\nINSTRUCTIONS\ngrault\nCONTENT\nabc"
        )
        document_mocks._read_document.assert_called_once_with(Path("bar"), "foo")

    def test_show_doc_ids_renders_pipeline_documents(
        self, documents: DocumentsFixture, mocker: MockerFixture
    ) -> None:
        render: MagicMock = mocker.patch.object(
            Documents, "_render", return_value="bar"
        )
        assert documents.show(["foo"], 3) == "bar"
        render.assert_called_once_with(Path("output/foo"), ["foo"], 3)


class TestTestSupplement:
    def test_test_supplement_file_generates_one_per_entry_and_reports(
        self,
        tmp_path: Path,
        documents: DocumentsFixture,
        constructor_mocks: ConstructorMocks,
        document_mocks: DocumentMocks,
        supplement_test_mocks: SupplementTestMocks,
    ) -> None:
        (tmp_path / "supplement.yaml").write_text(
            "supplements:\n"
            "  - name: foo\n"
            "    description: bar\n"
            "    count: 10\n"
            "    requirements:\n"
            "      baz: qux\n"
        )
        documents.test_supplement(tmp_path / "supplement.yaml", 5)
        supplement_test_mocks._reset_folder.assert_called_once_with(
            Documents.TEST_FOLDER
        )
        written: dict[Path, str] = {
            call.args[0]: call.args[1]
            for call in document_mocks._write_text.call_args_list
        }
        assert safe_load(written[Documents.TEST_FOLDER / "pipeline.yml"]) == {
            "output": {"subdirectory": "supplement"},
        }
        assert safe_load(written[Documents.TEST_FOLDER / "supplement.yaml"]) == {
            "supplements": [
                {
                    "name": "foo",
                    "description": "bar",
                    "count": 1,
                    "requirements": {"baz": "qux"},
                }
            ]
        }
        supplement_test_mocks._supplement.assert_called_once_with(Documents.TEST_FOLDER)
        supplement_test_mocks._doc_ids.assert_called_once_with(
            Documents.TEST_FOLDER / "output" / "supplement"
        )
        supplement_test_mocks._render.assert_called_once_with(
            Documents.TEST_FOLDER / "output" / "supplement", ["foo"], 5
        )
        constructor_mocks.getLogger.return_value.info.assert_called_once_with("bar")


class TestMain:
    @pytest.mark.parametrize(
        "arguments, method, expected",
        [
            (["sample", "--size", "5", "--batch", "2"], "sample", (5, 2)),
            (["show", "foo", "bar", "--chars", "10"], "show", (["foo", "bar"], 10)),
            (
                ["test-supplement", "baz.yaml"],
                "test_supplement",
                (Path("baz.yaml"), 4000),
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
