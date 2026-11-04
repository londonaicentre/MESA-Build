import json
import logging
import shutil
from argparse import ArgumentParser, Namespace
from difflib import unified_diff
from pathlib import Path
from random import Random

from docsynth.correct import Corrector
from docsynth.types.correction_rules import (
    DocumentCorrectionRule,
    DocumentCorrectionRules,
)
from docsynth.types.documents import DocsynthDocument
from yaml import safe_load


class Documents:
    """Compact views of a pipeline's generated documents, run from the pipeline
    folder, for authoring a document correction prompt.
    """

    SCRATCH_FOLDER: Path = Path("_scratch/corrections")
    TEST_FOLDER: Path = SCRATCH_FOLDER / "prompt_test"

    def __init__(self) -> None:
        self.__logger: logging.Logger = logging.getLogger(__name__)
        logging.basicConfig(format="%(message)s", level=logging.INFO)

        self.__documents_folder: Path = Path("output") / str(
            safe_load(Documents._read_text(Path("pipeline.yml")))["output"][
                "subdirectory"
            ]
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

    def _doc_ids(self) -> list[str]:
        return sorted(
            document_file.stem
            for document_file in self.__documents_folder.glob("*.json")
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
        doc_ids: list[str] = self._doc_ids()
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

    def show(self, doc_ids: list[str], chars: int) -> str:
        """Render documents compactly, with their prompt's instructions and
        truncated content.

        Args:
            doc_ids (list[str]): Documents to render
            chars (int): Maximum characters of each document's content

        Returns:
            str: The rendered documents

        """
        return "\n\n".join(
            f"=== {document.doc_id}\nINSTRUCTIONS\n{Documents.instructions(document)}"
            f"\nCONTENT\n{document.content[:chars]}"
            for doc_id in doc_ids
            for document in [Documents._read_document(self.__documents_folder, doc_id)]
        )

    def test_prompt(self, prompt_file: Path, doc_ids: list[str], chars: int) -> None:
        """Review and correct copies of some documents with a correction prompt,
        leaving the documents untouched, and report each document's rules and
        changes.

        Args:
            prompt_file (Path): Document correction prompt
            doc_ids (list[str]): Documents to review and correct
            chars (int): Maximum characters of each document's diff

        """
        rules_file: Path = Documents.TEST_FOLDER / "document_correction.yaml"
        Documents._reset_folder(Documents.TEST_FOLDER)
        before: dict[str, DocsynthDocument] = {
            doc_id: Documents._read_document(self.__documents_folder, doc_id)
            for doc_id in doc_ids
        }
        for doc_id, document in before.items():
            Documents._write_text(
                Documents.TEST_FOLDER / f"{doc_id}.json",
                document.model_dump_json(indent=2),
            )
        corrector: Corrector = Corrector(Documents.TEST_FOLDER, rules_file)
        corrector.review(prompt_file)
        corrector.correct_with_rules()
        corrector.correct_with_claude()
        rules: list[DocumentCorrectionRule] = DocumentCorrectionRules.load(
            rules_file
        ).rules
        self.__logger.info(
            "\n\n".join(
                f"=== {doc_id}\n"
                + "".join(
                    f"{rule.action}: {rule.description}\n"
                    for rule in rules
                    if rule.document == doc_id
                )
                + "\n".join(
                    list(
                        unified_diff(
                            document.content.splitlines(),
                            Documents._read_document(
                                Documents.TEST_FOLDER, doc_id
                            ).content.splitlines(),
                            lineterm="",
                            n=1,
                        )
                    )[2:]
                )[:chars]
                for doc_id, document in before.items()
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
        test_prompt_parser: ArgumentParser = commands.add_parser("test-prompt")
        test_prompt_parser.add_argument("prompt_file", type=Path)
        test_prompt_parser.add_argument("doc_ids", nargs="+")
        test_prompt_parser.add_argument("--chars", type=int, default=1500)
        arguments: Namespace = parser.parse_args()
        documents: Documents = Documents()
        match arguments.command:
            case "sample":
                documents.sample(arguments.size, arguments.batch)
            case "show":
                logging.getLogger(__name__).info(
                    documents.show(arguments.doc_ids, arguments.chars)
                )
            case "test-prompt":
                documents.test_prompt(
                    arguments.prompt_file, arguments.doc_ids, arguments.chars
                )


if __name__ == "__main__":
    Documents.main()
