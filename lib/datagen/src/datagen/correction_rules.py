import re
from pathlib import Path
from typing import Literal

import yaml
from jsonpath_ng import DatumInContext, JSONPath, parse  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from datagen.types import JSONValue, Scalar


class CorrectionRule(BaseModel):
    """A single declarative correction.

    Attributes:
        name (str): Unique rule name, used when reporting applications
        description (str): What the rule corrects and why
        path (str): JSONPath to the targets within a sample (e.g.
            `procedures[*].implants[*].manufacturer`)
        action (str): One of `clear` (set to null), `set` (set to `value`),
            `lower` (lowercase a string), `remove` (remove a list element)
            or `remove_items` (remove the items in `value` from a list)
        value (Scalar | list[Scalar]): Argument of `set` or `remove_items`.
            Defaults to None
        when (dict[str, JSONValue]): JSON Schema the rule's document must
            satisfy. Defaults to an empty schema, which always holds

    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    path: str
    action: Literal["clear", "set", "lower", "remove", "remove_items"]
    value: Scalar | list[Scalar] = None
    when: dict[str, JSONValue] = {}
    _expression: JSONPath = PrivateAttr()

    @model_validator(mode="after")
    def compile(self) -> "CorrectionRule":
        """Check that `when` is a valid JSON Schema and compile the path.

        Returns:
            CorrectionRule: The validated rule

        Raises:
            SchemaError: If `when` is not a valid JSON Schema
            JsonPathParserError: If `path` is not a valid JSONPath

        """
        Draft202012Validator.check_schema(self.when)
        self._expression = parse(self.path)
        return self

    @staticmethod
    def _substitute(node: JSONValue, value: JSONValue) -> JSONValue:
        if isinstance(node, str):
            return node.replace("{value}", re.escape(str(value)))
        if isinstance(node, dict):
            return {
                key: CorrectionRule._substitute(item, value)
                for key, item in node.items()
            }
        if isinstance(node, list):
            return [CorrectionRule._substitute(item, value) for item in node]
        return node

    def _holds(self, match: DatumInContext, note: str) -> bool:
        return Draft202012Validator(
            {
                key: CorrectionRule._substitute(schema, match.value)
                for key, schema in self.when.items()
            }
        ).is_valid(
            {
                "value": match.value,
                "parent": match.context.value if match.context else None,
                "note": note,
            }
        )

    def _transform(self, value: JSONValue) -> JSONValue:
        match self.action:
            case "clear":
                return None
            case "lower":
                return value.lower() if isinstance(value, str) else value
            case "remove_items":
                return (
                    [
                        item
                        for item in value
                        if item
                        not in (
                            self.value if isinstance(self.value, list) else [self.value]
                        )
                    ]
                    if isinstance(value, list)
                    else value
                )
            case _:
                return self.value

    def apply(
        self, sample: dict[str, JSONValue], note: str
    ) -> tuple[dict[str, JSONValue], bool]:
        """Apply the rule to a sample.

        Args:
            sample (dict[str, JSONValue]): Extracted sample
            note (str): Source note the sample was extracted from

        Returns:
            tuple[dict[str, JSONValue], bool]: The corrected sample, and
                whether the rule acted on any targets

        """
        matches: list[DatumInContext] = [
            match for match in self._expression.find(sample) if self._holds(match, note)
        ]
        for match in reversed(matches):
            if self.action == "remove":
                match.full_path.filter(lambda _: True, sample)
            else:
                match.full_path.update(sample, self._transform(match.value))
        return sample, bool(matches)


class CorrectionRules(BaseModel):
    """An ordered set of correction rules, applied in the order given.

    Attributes:
        rules (list[CorrectionRule]): The rules

    """

    rules: list[CorrectionRule]

    @staticmethod
    def load(rules_file: Path) -> "CorrectionRules":
        """Load and validate rules from a YAML file with a top-level `rules` list.

        Args:
            rules_file (Path): The rules file

        Returns:
            CorrectionRules: The validated rules

        """
        return CorrectionRules.model_validate(yaml.safe_load(rules_file.read_text()))

    def apply(
        self, sample: dict[str, JSONValue], note: str
    ) -> tuple[dict[str, JSONValue], bool]:
        """Apply every rule to a sample.

        Args:
            sample (dict[str, JSONValue]): Extracted sample
            note (str): Source note the sample was extracted from

        Returns:
            tuple[dict[str, JSONValue], bool]: The corrected sample, and
                whether any rule acted on it

        """
        changed: bool = False
        for rule in self.rules:
            sample, acted = rule.apply(sample, note)
            changed = changed or acted
        return sample, changed
