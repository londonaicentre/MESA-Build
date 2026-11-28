import json
import logging
import sys
from pathlib import Path
from typing import Final

import yaml
from mesa_types.base_model import BaseModelRecords, BaseModelReport
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


class BaseModelChecker:
    """Checks a base model's files for unsafe content, and records models that pass."""

    __SUFFIXES: Final = frozenset(
        {".safetensors", ".json", ".txt", ".md", ".jinja", ".model"}
    )
    __NAMES: Final = frozenset({".gitattributes", "LICENSE"})
    __PROMPT: Final = "Extract the diagnoses from: Patient presents with chest pain."
    __CONFIGS: Final = ("config.json", "tokenizer_config.json")
    __NETWORK_EVENTS: Final = frozenset(
        {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.sendmsg"}
    )

    @staticmethod
    def check(folder: Path, repo_id: str, revision: str) -> BaseModelReport:
        """Check a base model's file formats and remote code, then run a sample prompt in a model jail if they pass.

        Args:
            folder (Path): Folder holding the model's files.
            repo_id (str): Hugging Face repo id of the model.
            revision (str): Commit sha of the revision in the folder.

        Returns:
            BaseModelReport: The model, its revision and any findings.

        """
        return BaseModelReport(
            repo_id=repo_id,
            revision=revision,
            findings=BaseModelChecker._check_formats(folder)
            or BaseModelChecker._check_remote_code(folder)
            or BaseModelChecker._jail(folder),
        )

    @staticmethod
    def record(report: BaseModelReport, path: Path) -> None:
        """Add a model with no findings to a record of checked base models.

        Args:
            report (BaseModelReport): The report of the model's check.
            path (Path): YAML file holding the record.

        Raises:
            ValueError: If the report has findings.

        """
        if report.findings:
            raise ValueError(f"{report.repo_id} has findings, so cannot be recorded")
        records: BaseModelRecords = BaseModelRecords.model_validate(
            yaml.safe_load(path.read_text())
        )
        records.models[report.repo_id] = report.revision
        path.write_text(yaml.safe_dump(records.model_dump()))

    @staticmethod
    def _check_formats(folder: Path) -> list[str]:
        return [
            f"{path.relative_to(folder).as_posix()}: format not allowed"
            for path in sorted(folder.rglob("*"))
            if path.is_file()
            and path.suffix not in BaseModelChecker.__SUFFIXES
            and path.name not in BaseModelChecker.__NAMES
        ]

    @staticmethod
    def _check_remote_code(folder: Path) -> list[str]:
        return [
            f"{name}: needs remote code (auto_map)"
            for name in BaseModelChecker.__CONFIGS
            if (folder / name).is_file()
            and "auto_map" in json.loads((folder / name).read_text())
        ]

    @staticmethod
    def _jail(folder: Path) -> list[str]:
        events: list[str] = []
        watching: bool = True

        def audit(event: str, arguments: tuple[object, ...]) -> None:
            if watching and event in BaseModelChecker.__NETWORK_EVENTS:
                logger.warning(
                    f"Network access from model in jail: {event} {arguments}"
                )
                events.append(f"model jail: network access ({event} {arguments})")
                raise PermissionError(event)

        sys.addaudithook(audit)
        try:
            BaseModelChecker._generate(folder)
        finally:
            watching = False
        return events

    @staticmethod
    def _generate(folder: Path) -> str:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(folder)
        inputs: object = tokenizer.apply_chat_template(
            [{"role": "user", "content": BaseModelChecker.__PROMPT}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        assert isinstance(inputs, BatchEncoding)
        return str(
            tokenizer.decode(
                AutoModelForCausalLM.from_pretrained(
                    folder, use_safetensors=True
                ).generate(**inputs, max_new_tokens=8)[0]
            )
        )
