import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from datagen.correction_rules import CorrectionRules
from datagen.extraction import extract_and_validate_json
from datagen.types import JSONValue
from utils.llm import BatchOutput


class BatchOutputCorrector:
    """Corrector of Bedrock batch outputs, overwriting the output file in place.

    Args:
        schema (type[BaseModel]): Schema that outputs must validate against
        batch_output_file (Path): Bedrock batch output file to correct

    """

    __THINKING_TOKENS: int = 3000

    def __init__(self, schema: type[BaseModel], batch_output_file: Path) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        if not self._logger.hasHandlers():
            self._logger.addHandler(logging.StreamHandler())
            self._logger.setLevel(logging.INFO)

        self.__schema: type[BaseModel] = schema
        self.__batch_output_file: Path = batch_output_file

    def _read(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in self.__batch_output_file.read_text().splitlines()
        ]

    def _write(self, records: list[dict[str, object]]) -> None:
        temporary_file: Path = self.__batch_output_file.with_suffix(".tmp")
        temporary_file.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        os.replace(temporary_file, self.__batch_output_file)

    @staticmethod
    def _get_note(record: dict[str, object]) -> str:
        return BatchOutput.model_validate(record).modelInput.messages[0].content[0].text

    @staticmethod
    def _get_response(record: dict[str, object]) -> str:
        return "".join(
            block.text
            for block in BatchOutput.model_validate(record).modelOutput.content
        )

    @staticmethod
    def _replace_response(
        record: dict[str, object], response: str
    ) -> dict[str, object]:
        model_output: object = record["modelOutput"]
        if not isinstance(model_output, dict):
            raise ValueError(f"record {record.get('recordId')} has no modelOutput")
        return {
            **record,
            "modelOutput": {
                **model_output,
                "content": [{"type": "text", "text": response}],
            },
        }

    @staticmethod
    def _run_claude(system_prompt: str, user_prompt: str) -> str:
        return subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                "haiku",
                "--system-prompt",
                system_prompt,
                "--tools",
                "",
                "--strict-mcp-config",
                "--setting-sources",
                "",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--settings",
                '{"alwaysThinkingEnabled": true}',
            ],
            input=user_prompt,
            capture_output=True,
            text=True,
            check=True,
            cwd=tempfile.gettempdir(),
            env={
                **os.environ,
                "MAX_THINKING_TOKENS": str(BatchOutputCorrector.__THINKING_TOKENS),
            },
        ).stdout.strip()

    def correct_with_claude(self, prompt_file: Path, extraction_prompt: str) -> int:
        """Send, to Claude Code, every output that has no valid extraction.

        Args:
            prompt_file (Path): Correction prompt containing an
                `{EXTRACTION_PROMPT}` placeholder
            extraction_prompt (str): Prompt originally used to generate the
                outputs

        Returns:
            int: Number of records corrected

        """
        system_prompt: str = prompt_file.read_text().replace(
            "{EXTRACTION_PROMPT}", extraction_prompt
        )
        records: list[dict[str, object]] = self._read()
        corrected: int = 0
        for index, record in enumerate(records):
            if (
                extract_and_validate_json(
                    BatchOutputCorrector._get_response(record), self.__schema
                )[0]
                is not None
            ):
                continue
            self._logger.info(
                f"Correcting record {index + 1}/{len(records)} ({corrected} done)"
            )
            records[index] = BatchOutputCorrector._replace_response(
                record,
                BatchOutputCorrector._run_claude(
                    system_prompt,
                    f"<note>\n{BatchOutputCorrector._get_note(record)}\n</note>\n\n"
                    f"<original_response>\n{BatchOutputCorrector._get_response(record)}\n</original_response>",
                ),
            )
            self._write(records)
            corrected += 1
        return corrected

    def correct_with_rules(self, rules_file: Path) -> int:
        """Apply declarative rules to every output that has a valid extraction.

        Args:
            rules_file (Path): YAML rules file, as read by `CorrectionRules`

        Returns:
            int: Number of records corrected

        Raises:
            ValidationError: If a corrected output no longer validates against
                the schema

        """
        rules: CorrectionRules = CorrectionRules.load(rules_file)
        records: list[dict[str, object]] = self._read()
        corrected: int = 0
        for index, record in enumerate(records):
            validated, _ = extract_and_validate_json(
                BatchOutputCorrector._get_response(record), self.__schema
            )
            if validated is None:
                continue
            sample: dict[str, JSONValue] = validated.model_dump(mode="json")
            sample, changed = rules.apply(
                sample, BatchOutputCorrector._get_note(record)
            )
            if changed:
                records[index] = BatchOutputCorrector._replace_response(
                    record,
                    "<output>\n"
                    + self.__schema.model_validate(sample).model_dump_json(indent=2)
                    + "\n</output>",
                )
                corrected += 1
        self._write(records)
        return corrected
