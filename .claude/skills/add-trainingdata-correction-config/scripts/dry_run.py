import json
import logging
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from difflib import unified_diff
from pathlib import Path
from random import Random
from typing import NamedTuple

from corpus import Corpus
from datagen.correction_rules import CorrectionRule, CorrectionRules
from datagen.extraction import extract_and_validate_json
from datagen.types import JSONValue


class Change(NamedTuple):
    record_id: str
    note: str
    before: dict[str, JSONValue]
    after: dict[str, JSONValue]


class DryRun:
    """Apply correction rules to a pipeline's batch outputs without writing
    them, run from the pipeline folder, reporting what each rule would change.

    Args:
        schema_package (str): Schema package the pipeline extracts with (e.g.
            `opnoteschema`)
        rules_file (Path): YAML rules file, as read by `CorrectionRules`

    """

    def __init__(self, schema_package: str, rules_file: Path) -> None:
        self.__corpus: Corpus = Corpus(schema_package)
        self.__logger: logging.Logger = logging.getLogger(__name__)
        self.__rules: CorrectionRules = CorrectionRules.load(rules_file)

    def _apply(
        self, rules: list[CorrectionRule]
    ) -> tuple[dict[str, list[Change]], list[str]]:
        changes: dict[str, list[Change]] = {rule.name: [] for rule in rules}
        unstable: list[str] = []
        for _, record in Corpus.records():
            validated, _ = extract_and_validate_json(
                Corpus.response(record), self.__corpus.get_schema()
            )
            if validated is None:
                continue
            note: str = Corpus.note(record)
            sample: dict[str, JSONValue] = validated.model_dump(mode="json")
            for rule in rules:
                before: dict[str, JSONValue] = deepcopy(sample)
                sample, acted = rule.apply(sample, note)
                if acted:
                    changes[rule.name].append(
                        Change(record.recordId, note, before, deepcopy(sample))
                    )
            self.__corpus.get_schema().model_validate(sample)
            if any(rule.apply(sample, note)[1] for rule in rules):
                unstable.append(record.recordId)
        return changes, unstable

    @staticmethod
    def _render(rule_name: str, change: Change, chars: int) -> str:
        return (
            f"=== {rule_name}: {change.record_id}\n"
            + "\n".join(
                list(
                    unified_diff(
                        json.dumps(change.before, indent=1).splitlines(),
                        json.dumps(change.after, indent=1).splitlines(),
                        lineterm="",
                        n=1,
                    )
                )[2:]
            )
            + f"\nNOTE\n{change.note[:chars]}"
        )

    def run(self, rule_name: str | None, show: int, chars: int) -> None:
        """Report the records each rule would change, whether a second pass
        would change any further, and a random sample of the changes.

        Args:
            rule_name (str | None): Only apply this rule, or None for every rule
            show (int): Number of changes to show per rule
            chars (int): Maximum characters of each note shown

        Raises:
            ValueError: If no rule has the name given
            ValidationError: If a corrected sample no longer validates against
                the schema

        """
        rules: list[CorrectionRule] = [
            rule for rule in self.__rules.rules if rule_name in (None, rule.name)
        ]
        if not rules:
            raise ValueError(f"No rule named {rule_name}")
        changes, unstable = self._apply(rules)
        self.__logger.info(
            "\n".join(
                f"{name}: {len(rule_changes)}" for name, rule_changes in changes.items()
            )
            + "\nRecords changed: "
            + str(
                len(
                    {
                        change.record_id
                        for rule_changes in changes.values()
                        for change in rule_changes
                    }
                )
            )
            + f"\nRecords changed again by a second pass: {len(unstable)} {unstable[:5]}"
        )
        if show:
            self.__logger.info(
                "\n\n".join(
                    DryRun._render(name, change, chars)
                    for name, rule_changes in changes.items()
                    for change in Random().sample(
                        rule_changes, min(show, len(rule_changes))
                    )
                )
            )

    @staticmethod
    def main() -> None:
        """Run the dry run given on the command line."""
        parser: ArgumentParser = ArgumentParser(description=DryRun.__doc__)
        parser.add_argument("--schema", required=True)
        parser.add_argument("rules_file", type=Path)
        parser.add_argument("--rule")
        parser.add_argument("--show", type=int, default=0)
        parser.add_argument("--chars", type=int, default=600)
        arguments: Namespace = parser.parse_args()
        DryRun(arguments.schema, arguments.rules_file).run(
            arguments.rule, arguments.show, arguments.chars
        )


if __name__ == "__main__":
    DryRun.main()
