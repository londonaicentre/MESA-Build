import json
import logging
import shutil
from argparse import ArgumentParser, Namespace
from contextlib import chdir
from pathlib import Path
from random import Random

from docsynth.generate import Generator
from docsynth.types.documents import DocsynthDocument
from docsynth.types.supplements import Supplements
from yaml import safe_dump, safe_load


class Documents:
    """Compact views of a pipeline's generated documents, run from the pipeline
    folder, for authoring a document supplement.
    """

    SCRATCH_FOLDER: Path = Path("_scratch/corrections")
    TEST_FOLDER: Path = SCRATCH_FOLDER / "supplement_test"

    def __init__(self) -> None:
        self.__logger: logging.Logger = logging.getLogger(__name__)
        logging.basicConfig(format="%(message)s", level=logging.INFO)

        self.__pipeline: dict[str, dict[str, str]] = safe_load(
            Documents._read_text(Path("pipeline.yml"))
        )
        self.__documents_folder: Path = Path("output") / str(
            self.__pipeline["output"]["subdirectory"]
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text()

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    @staticmethod
    def _reset_folder(folder: Path) -> None:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    @staticmethod
    def _supplement(folder: Path) -> int:
        with chdir(folder):
            return Generator().supplement(Path("supplement.yaml"))

    @staticmethod
    def _doc_ids(folder: Path) -> list[str]:
        return sorted(
            document_file.stem
            for document_file in folder.glob("*.json")
            if document_file.name != "metadata.json"
        )

    @staticmethod
    def _read_document(folder: Path, doc_id: str) -> DocsynthDocument:
        return DocsynthDocument.model_validate_json(
            Documents._read_text(folder / f"{doc_id}.json")
        )

    @staticmethod
    def instructions(document: DocsynthDocument) -> str:
        """Get the style, content and profile sections of a document's prompt.

        Args:
            document (DocsynthDocument): The document

        Returns:
            str: The sections, without the example structure

        """
        return "\n## ".join(
            section
            for section in document.prompt.split("# SPECIFIC INSTRUCTIONS")[1]
            .split("# YOUR TASK")[0]
            .split("\n## ")
            if not section.startswith("MIMIC")
        ).strip()

    def sample(self, size: int, batch: int) -> list[Path]:
        """Write random batch manifests of doc IDs, for fan-out.

        Args:
            size (int): Number of documents to sample
            batch (int): Number of documents per manifest

        Returns:
            list[Path]: The manifest files

        """
        doc_ids: list[str] = Documents._doc_ids(self.__documents_folder)
        sampled: list[str] = Random().sample(doc_ids, min(size, len(doc_ids)))
        manifests: list[Path] = [
            Documents.SCRATCH_FOLDER / f"documents_{index // batch + 1}.json"
            for index in range(0, len(sampled), batch)
        ]
        for index, manifest in enumerate(manifests):
            Documents._write_text(
                manifest, json.dumps(sampled[index * batch : (index + 1) * batch])
            )
        self.__logger.info("\n".join(str(manifest) for manifest in manifests))
        return manifests

    @staticmethod
    def _render(folder: Path, doc_ids: list[str], chars: int) -> str:
        return "\n\n".join(
            f"=== {document.doc_id}\nINSTRUCTIONS\n{Documents.instructions(document)}"
            f"\nCONTENT\n{document.content[:chars]}"
            for doc_id in doc_ids
            for document in [Documents._read_document(folder, doc_id)]
        )

    def show(self, doc_ids: list[str], chars: int) -> str:
        """Render documents compactly, with their prompt's instructions and
        truncated content.

        Args:
            doc_ids (list[str]): Documents to render
            chars (int): Maximum characters of each document's content

        Returns:
            str: The rendered documents

        """
        return Documents._render(self.__documents_folder, doc_ids, chars)

    def test_supplement(self, supplement_file: Path, chars: int) -> None:
        """Generate one document per supplement entry, apart from the pipeline's.

        Args:
            supplement_file (Path): Supplement file
            chars (int): Maximum characters of each document's content

        """
        Documents._reset_folder(Documents.TEST_FOLDER)
        Documents._write_text(
            Documents.TEST_FOLDER / "pipeline.yml",
            safe_dump(
                self.__pipeline
                | {"output": self.__pipeline["output"] | {"subdirectory": "supplement"}}
            ),
        )
        Documents._write_text(
            Documents.TEST_FOLDER / "supplement.yaml",
            safe_dump(
                Supplements(
                    supplements=[
                        supplement.model_copy(update={"count": 1})
                        for supplement in Supplements.load(supplement_file).supplements
                    ]
                ).model_dump()
            ),
        )
        Documents._supplement(Documents.TEST_FOLDER)
        self.__logger.info(
            Documents._render(
                Documents.TEST_FOLDER / "output" / "supplement",
                Documents._doc_ids(Documents.TEST_FOLDER / "output" / "supplement"),
                chars,
            )
        )

    @staticmethod
    def main() -> None:
        """Run the command given on the command line."""
        parser: ArgumentParser = ArgumentParser(description=Documents.__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        sample_parser: ArgumentParser = commands.add_parser("sample")
        sample_parser.add_argument("--size", type=int, default=100)
        sample_parser.add_argument("--batch", type=int, default=25)
        show_parser: ArgumentParser = commands.add_parser("show")
        show_parser.add_argument("doc_ids", nargs="+")
        show_parser.add_argument("--chars", type=int, default=4000)
        test_supplement_parser: ArgumentParser = commands.add_parser("test-supplement")
        test_supplement_parser.add_argument("supplement_file", type=Path)
        test_supplement_parser.add_argument("--chars", type=int, default=4000)
        arguments: Namespace = parser.parse_args()
        documents: Documents = Documents()
        match arguments.command:
            case "sample":
                documents.sample(arguments.size, arguments.batch)
            case "show":
                logging.getLogger(__name__).info(
                    documents.show(arguments.doc_ids, arguments.chars)
                )
            case "test-supplement":
                documents.test_supplement(arguments.supplement_file, arguments.chars)


if __name__ == "__main__":
    Documents.main()
