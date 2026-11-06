import json
import logging
import shutil
from argparse import ArgumentParser, Namespace
from contextlib import chdir
from pathlib import Path
from random import Random

from docsynth.generate import Generator
from docsynth.types.documents import DocsynthDocument
from docsynth.types.sampling import SamplingAttribute, SamplingElement
from docsynth.types.supplements import Supplements
from docsynth.types.wrapper import DocsynthAssets
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
        self.__documents_folder: Path = Path("data/docsynth/documents") / str(
            self.__pipeline["output"]["subdirectory"]
        )
        self.__sections: dict[str, SamplingElement] = Documents._sections(
            self.__pipeline["profile_selection"]["domain"]
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
    def _sections(domain: str) -> dict[str, SamplingElement]:
        assets: DocsynthAssets = DocsynthAssets.from_domain(domain)
        return assets.load_style_data().root | assets.load_content_data().root

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

    def _options(self, document: DocsynthDocument) -> dict[str, list[str]]:
        instructions: str = Documents.instructions(document)
        options: dict[str, list[str]] = {}
        for section_name, element in self.__sections.items():
            header: str = f"**{section_name.replace('_', ' ').title()}:**\n"
            items: list[str] = [
                item.strip()
                for item in (
                    "\n" + instructions.split(header)[1].split("\n\n")[0]
                    if header in instructions
                    else ""
                ).split("\n- ")[1:]
            ]
            options[section_name] = [
                option
                for option, attribute in element.get_options().items()
                if attribute.description and attribute.description.strip() in items
            ]
            if element.mutually_exclusive and not options[section_name]:
                options[section_name] = [
                    option
                    for option, attribute in element.get_options().items()
                    if not attribute.description
                ]
        return options

    def census(self) -> None:
        """Report each option's sampled share against its configured probability."""
        doc_ids: list[str] = Documents._doc_ids(self.__documents_folder)
        options: list[dict[str, list[str]]] = [
            self._options(Documents._read_document(self.__documents_folder, doc_id))
            for doc_id in doc_ids
        ]
        lines: list[str] = []
        for section_name, element in self.__sections.items():
            attributes: dict[str, SamplingAttribute] = element.get_options()
            total: float = (
                sum(attribute.probability for attribute in attributes.values())
                if element.mutually_exclusive
                else 1
            )
            lines.append(section_name)
            lines.extend(
                f"  {option}: {count}/{len(doc_ids)}"
                f" ({count / max(len(doc_ids), 1):.0%}),"
                f" configured {attribute.probability / total:.0%}"
                for option, attribute in attributes.items()
                for count in [
                    sum(
                        option in document_options[section_name]
                        for document_options in options
                    )
                ]
            )
        self.__logger.info("\n".join(lines))

    def sample(
        self, size: int, batch: int, requirement: tuple[str, str] | None = None
    ) -> list[Path]:
        """Write random batch manifests of doc IDs, for fan-out.

        Args:
            size (int): Number of documents to sample
            batch (int): Number of documents per manifest
            requirement (tuple[str, str], optional): Section and option the
                sampled documents' prompts must have selected. Defaults to
                None, which samples every document

        Returns:
            list[Path]: The manifest files

        Raises:
            ValueError: If the requirement names an unknown section or option

        """
        doc_ids: list[str] = Documents._doc_ids(self.__documents_folder)
        if requirement is not None:
            section_name, option = requirement
            if (
                section_name not in self.__sections
                or option not in self.__sections[section_name].get_options()
            ):
                raise ValueError(f"Unknown requirement: {section_name}={option}")
            doc_ids = [
                doc_id
                for doc_id in doc_ids
                if option
                in self._options(
                    Documents._read_document(self.__documents_folder, doc_id)
                )[section_name]
            ]
        sampled: list[str] = Random().sample(doc_ids, min(size, len(doc_ids)))
        manifests: list[Path] = [
            Documents.SCRATCH_FOLDER
            / f"{'-'.join(['documents', *(requirement or ())])}_{index // batch + 1}.json"
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
                Documents.TEST_FOLDER / "data/docsynth/documents/supplement",
                Documents._doc_ids(
                    Documents.TEST_FOLDER / "data/docsynth/documents/supplement"
                ),
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
        sample_parser.add_argument(
            "--requirement", type=lambda value: tuple(value.split("=", 1))
        )
        commands.add_parser("census")
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
                documents.sample(arguments.size, arguments.batch, arguments.requirement)
            case "census":
                documents.census()
            case "show":
                logging.getLogger(__name__).info(
                    documents.show(arguments.doc_ids, arguments.chars)
                )
            case "test-supplement":
                documents.test_supplement(arguments.supplement_file, arguments.chars)


if __name__ == "__main__":
    Documents.main()
