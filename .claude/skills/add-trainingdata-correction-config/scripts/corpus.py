import json
import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from random import Random

from pydantic import BaseModel

from datagen.batch_corrector import BatchOutputCorrector
from datagen.extraction import extract_and_validate_json
from utils.llm import BatchOutput


class Corpus:
    """Compact views of a pipeline's batch outputs, run from the pipeline
    folder, for authoring correction config.

    Args:
        schema_package (str): Schema package the pipeline extracts with (e.g.
            `opnoteschema`)

    """

    BATCH_OUTPUT_FILE: Path = Path("anthropic_batch_job.jsonl.out")
    SCRATCH_FOLDER: Path = Path("_scratch/corrections")
    CENSUS_FILE: Path = SCRATCH_FOLDER / "census.json"

    def __init__(self, schema_package: str) -> None:
        self.__logger: logging.Logger = logging.getLogger(__name__)
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        logging.getLogger("datagen.extraction").setLevel(logging.CRITICAL)

        self.__schema_package: str = schema_package
        self.__schema: type[BaseModel] = import_module(schema_package).Schema
        Corpus.SCRATCH_FOLDER.mkdir(parents=True, exist_ok=True)

    def get_schema(self) -> type[BaseModel]:
        """Get the schema outputs must validate against.

        Returns:
            type[BaseModel]: The schema

        """
        return self.__schema

    @staticmethod
    def records(
        record_ids: set[str] | None = None,
    ) -> Iterator[tuple[str, BatchOutput]]:
        """Stream batch output records, without loading the whole file.

        Args:
            record_ids (set[str] | None): Only stream these records. Defaults
                to None, which streams every record

        Yields:
            tuple[str, BatchOutput]: The raw record line, and the parsed record

        """
        with Corpus.BATCH_OUTPUT_FILE.open() as batch_output_file:
            yield from (
                (line, record)
                for line in batch_output_file
                if record_ids is None
                or any(record_id in line for record_id in record_ids)
                for record in [BatchOutput.model_validate_json(line)]
                if record_ids is None or record.recordId in record_ids
            )

    @staticmethod
    def note(record: BatchOutput) -> str:
        """Get the source note of a record.

        Args:
            record (BatchOutput): The record

        Returns:
            str: The note

        """
        return record.modelInput.messages[0].content[0].text

    @staticmethod
    def response(record: BatchOutput) -> str:
        """Get the model response of a record.

        Args:
            record (BatchOutput): The record

        Returns:
            str: The response

        """
        return "".join(block.text for block in record.modelOutput.content)

    def status(self, record: BatchOutput) -> str:
        """Classify a record by whether, and how, its extraction failed.

        Args:
            record (BatchOutput): The record

        Returns:
            str: One of `valid`, `invalid` (JSON failing the schema), `empty`,
                `max_tokens`, `has_output` (an `<output>` tag without JSON) or
                `prose`

        """
        response: str = Corpus.response(record)
        validated, data = extract_and_validate_json(response, self.__schema)
        if validated is not None:
            return "valid"
        if data is not None:
            return "invalid"
        if not response.strip():
            return "empty"
        if record.modelOutput.stop_reason == "max_tokens":
            return "max_tokens"
        return "has_output" if "<output>" in response else "prose"

    def census(self) -> dict[str, list[str]]:
        """Group record IDs by extraction status, writing them to the census file.

        Returns:
            dict[str, list[str]]: Record IDs by status

        """
        record_ids_by_status: dict[str, list[str]] = {}
        for _, record in Corpus.records():
            record_ids_by_status.setdefault(self.status(record), []).append(
                record.recordId
            )
        Corpus.CENSUS_FILE.write_text(json.dumps(record_ids_by_status))
        self.__logger.info(
            "\n".join(
                f"{status}: {len(record_ids)}"
                for status, record_ids in record_ids_by_status.items()
            )
            + f"\nRecord IDs by status written to {Corpus.CENSUS_FILE}"
        )
        return record_ids_by_status

    def sample(self, status: str, size: int, batch: int) -> list[Path]:
        """Write random batch manifests of record IDs, for fan-out.

        Args:
            status (str): A census status
            size (int): Number of items to sample
            batch (int): Number of items per manifest

        Returns:
            list[Path]: The manifest files

        Raises:
            RuntimeError: If the batch output changed after the census was taken

        """
        if (
            Corpus.BATCH_OUTPUT_FILE.stat().st_mtime
            > Corpus.CENSUS_FILE.stat().st_mtime
        ):
            raise RuntimeError(
                f"{Corpus.BATCH_OUTPUT_FILE} changed after the census, so re-run census"
            )
        record_ids: list[str] = json.loads(Corpus.CENSUS_FILE.read_text())[status]
        sampled: list[str] = Random().sample(record_ids, min(size, len(record_ids)))
        manifests: list[Path] = [
            Corpus.SCRATCH_FOLDER / f"{status}_{index // batch + 1}.json"
            for index in range(0, len(sampled), batch)
        ]
        for index, manifest in enumerate(manifests):
            manifest.write_text(
                json.dumps(sampled[index * batch : (index + 1) * batch])
            )
        self.__logger.info("\n".join(str(manifest) for manifest in manifests))
        return manifests

    def show(self, record_ids: list[str], chars: int) -> str:
        """Render records compactly, with truncated text.

        Args:
            record_ids (list[str]): Records to render
            chars (int): Maximum characters of each note and response

        Returns:
            str: The rendered records

        """
        return "\n\n".join(
            f"=== {record.recordId} ({self.status(record)})\nNOTE\n"
            f"{Corpus.note(record)[:chars]}\nRESPONSE\n{Corpus.response(record)[:chars]}"
            for _, record in Corpus.records(set(record_ids))
        )

    def test_prompt(self, prompt_file: Path, record_ids: list[str], chars: int) -> None:
        """Test a correction prompt on copies of some records, leaving the batch
        output untouched, and report each record's status before and after.

        Args:
            prompt_file (Path): Correction prompt containing an
                `{EXTRACTION_PROMPT}` placeholder
            record_ids (list[str]): Records to correct
            chars (int): Maximum characters of each corrected response

        """
        test_file: Path = Corpus.SCRATCH_FOLDER / "prompt_test.jsonl.out"
        lines: list[tuple[str, BatchOutput]] = list(Corpus.records(set(record_ids)))
        test_file.write_text("".join(line for line, _ in lines))
        before: dict[str, str] = {
            record.recordId: self.status(record) for _, record in lines
        }
        BatchOutputCorrector(self.__schema, test_file).correct_with_claude(
            prompt_file,
            import_module(self.__schema_package + ".prompt_builder")
            .PromptBuilder()
            .build_datagen_prompt(),
        )
        self.__logger.info(
            "\n\n".join(
                f"=== {record.recordId}: {before[record.recordId]} -> {self.status(record)}\n"
                + Corpus.response(record)[:chars]
                for line in test_file.read_text().splitlines()
                for record in [BatchOutput.model_validate_json(line)]
            )
        )

    @staticmethod
    def main() -> None:
        """Run the command given on the command line."""
        parser: ArgumentParser = ArgumentParser(description=Corpus.__doc__)
        parser.add_argument("--schema", required=True)
        commands = parser.add_subparsers(dest="command", required=True)
        commands.add_parser("census")
        sample_parser: ArgumentParser = commands.add_parser("sample")
        sample_parser.add_argument("status")
        sample_parser.add_argument("--size", type=int, default=100)
        sample_parser.add_argument("--batch", type=int, default=25)
        show_parser: ArgumentParser = commands.add_parser("show")
        show_parser.add_argument("record_ids", nargs="+")
        show_parser.add_argument("--chars", type=int, default=4000)
        test_prompt_parser: ArgumentParser = commands.add_parser("test-prompt")
        test_prompt_parser.add_argument("prompt_file", type=Path)
        test_prompt_parser.add_argument("record_ids", nargs="+")
        test_prompt_parser.add_argument("--chars", type=int, default=1500)
        arguments: Namespace = parser.parse_args()
        corpus: Corpus = Corpus(arguments.schema)
        match arguments.command:
            case "census":
                corpus.census()
            case "sample":
                corpus.sample(
                    arguments.status,
                    arguments.size,
                    arguments.batch,
                )
            case "show":
                logging.getLogger(__name__).info(
                    corpus.show(arguments.record_ids, arguments.chars)
                )
            case "test-prompt":
                corpus.test_prompt(
                    arguments.prompt_file, arguments.record_ids, arguments.chars
                )


if __name__ == "__main__":
    Corpus.main()
