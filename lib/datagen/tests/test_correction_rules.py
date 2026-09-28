from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from jsonpath_ng import DatumInContext, parse  # type: ignore[import-untyped]
from jsonpath_ng.exceptions import JsonPathParserError  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError
from pydantic import ValidationError
from pytest_mock import MockerFixture

from datagen.correction_rules import CorrectionRule, CorrectionRules
from datagen.types import JSONValue


class CorrectionRuleFixture(CorrectionRule):
    @staticmethod
    def substitute(node: JSONValue, value: JSONValue) -> JSONValue:
        return CorrectionRule._substitute(node, value)

    def holds(self, match: DatumInContext, note: str) -> bool:
        return self._holds(match, note)

    def transform(self, value: JSONValue) -> JSONValue:
        return self._transform(value)


@dataclass
class HoldsMocks:
    _substitute: MagicMock


@dataclass
class ApplyMocks:
    _holds: MagicMock
    _transform: MagicMock


@pytest.fixture
def holds_mocks(mocker: MockerFixture) -> HoldsMocks:
    return HoldsMocks(
        _substitute=mocker.patch.object(
            CorrectionRule, "_substitute", side_effect=lambda node, _: node
        )
    )


@pytest.fixture
def apply_mocks(mocker: MockerFixture) -> ApplyMocks:
    return ApplyMocks(
        _holds=mocker.patch.object(CorrectionRule, "_holds", autospec=True),
        _transform=mocker.patch.object(
            CorrectionRule, "_transform", autospec=True, return_value="baz"
        ),
    )


def make_rule(**overrides: object) -> CorrectionRuleFixture:
    return CorrectionRuleFixture.model_validate(
        {
            "name": "foo",
            "description": "bar",
            "path": "baz",
            "action": "clear",
            **overrides,
        }
    )


class TestCorrectionRuleValidation:
    @pytest.mark.parametrize(
        "overrides, error",
        [
            ({"foo": "bar"}, ValidationError),
            ({"when": {"type": "foo"}}, SchemaError),
            ({"path": "foo[*"}, JsonPathParserError),
        ],
    )
    def test_compile_invalid_rule_raises(
        self, overrides: dict[str, object], error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            make_rule(**overrides)

    def test_compile_valid_rule_returns_rule(self) -> None:
        assert make_rule(when={"properties": {"value": {"const": "foo"}}}).name == "foo"


class TestCorrectionRuleSubstitute:
    @pytest.mark.parametrize(
        "node, expected",
        [
            ("(?i){value}", r"(?i)foo\.bar"),
            (
                {"pattern": "{value}", "items": ["x{value}", 1]},
                {"pattern": r"foo\.bar", "items": [r"xfoo\.bar", 1]},
            ),
            (1, 1),
        ],
    )
    def test_substitute_node_replaces_placeholders_with_escaped_value(
        self, node: JSONValue, expected: JSONValue
    ) -> None:
        assert CorrectionRuleFixture.substitute(node, "foo.bar") == expected


class TestCorrectionRuleHolds:
    @pytest.mark.parametrize(
        "path, sample, when, expected",
        [
            ("foo", {"foo": "bar"}, {}, True),
            ("foo", {"foo": "bar"}, {"properties": {"value": {"const": "bar"}}}, True),
            ("foo", {"foo": "bar"}, {"properties": {"value": {"const": "baz"}}}, False),
            (
                "foo",
                {"foo": 1, "qux": None},
                {"properties": {"parent": {"properties": {"qux": {"type": "null"}}}}},
                True,
            ),
            (
                "foo[*]",
                {"foo": [1]},
                {"properties": {"parent": {"minItems": 2}}},
                False,
            ),
            ("foo", {"foo": 1}, {"properties": {"note": {"pattern": "^qu"}}}, True),
            ("foo", {"foo": 1}, {"properties": {"note": {"pattern": "^ba"}}}, False),
        ],
    )
    def test_holds_document_returns_schema_validity(
        self,
        holds_mocks: HoldsMocks,
        path: str,
        sample: dict[str, JSONValue],
        when: dict[str, JSONValue],
        expected: bool,
    ) -> None:
        rule: CorrectionRuleFixture = make_rule(path=path, when=when)
        assert rule.holds(parse(path).find(sample)[0], "quux") is expected

    def test_holds_substitutes_target_value_into_schema(
        self, holds_mocks: HoldsMocks
    ) -> None:
        rule: CorrectionRuleFixture = make_rule(
            when={"properties": {"note": {"pattern": "{value}"}}}
        )
        rule.holds(parse("baz").find({"baz": "qux"})[0], "quux")
        holds_mocks._substitute.assert_called_once_with(
            {"note": {"pattern": "{value}"}}, "qux"
        )

    def test_holds_match_without_context_checks_null_parent(
        self, holds_mocks: HoldsMocks
    ) -> None:
        rule: CorrectionRuleFixture = make_rule(
            when={"properties": {"parent": {"type": "null"}}}
        )
        assert rule.holds(DatumInContext("foo"), "quux")


class TestCorrectionRuleTransform:
    @pytest.mark.parametrize(
        "overrides, value, expected",
        [
            ({}, "foo", None),
            ({"action": "set", "value": "bar"}, "foo", "bar"),
            ({"action": "lower"}, "FoO", "foo"),
            ({"action": "lower"}, 1, 1),
            ({"action": "remove_items", "value": ["bar"]}, ["foo", "bar"], ["foo"]),
            ({"action": "remove_items", "value": ["bar"]}, "bar", "bar"),
            ({"action": "remove_items", "value": "bar"}, ["foo", "bar"], ["foo"]),
        ],
    )
    def test_transform_action_returns_expected(
        self, overrides: dict[str, object], value: JSONValue, expected: JSONValue
    ) -> None:
        assert make_rule(**overrides).transform(value) == expected


class TestCorrectionRuleApply:
    def test_apply_field_path_updates_held_targets(
        self, apply_mocks: ApplyMocks
    ) -> None:
        apply_mocks._holds.side_effect = [True, False]
        sample: dict[str, JSONValue] = {"foo": [{"bar": 1}, {"bar": 2}]}
        result, acted = make_rule(path="foo[*].bar").apply(sample, "quux")
        assert acted is True
        assert result == {"foo": [{"bar": "baz"}, {"bar": 2}]}
        assert apply_mocks._holds.call_args_list[0].args[2] == "quux"

    def test_apply_remove_path_removes_held_elements(
        self, apply_mocks: ApplyMocks
    ) -> None:
        apply_mocks._holds.side_effect = [True, False, True]
        sample: dict[str, JSONValue] = {"foo": ["bar", "baz", "qux"]}
        result, acted = make_rule(action="remove", path="foo[*]").apply(sample, "quux")
        assert acted is True
        assert result == {"foo": ["baz"]}
        apply_mocks._transform.assert_not_called()

    def test_apply_unmatched_path_leaves_sample(self, apply_mocks: ApplyMocks) -> None:
        sample: dict[str, JSONValue] = {"foo": None}
        result, acted = make_rule(path="foo[*].bar").apply(sample, "quux")
        assert acted is False
        assert result == {"foo": None}
        apply_mocks._holds.assert_not_called()


class TestCorrectionRules:
    def test_load_yaml_file_returns_validated_rules(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch.object(
            Path,
            "read_text",
            return_value="rules:\n  - {name: foo, description: bar, path: baz, action: clear}\n",
        )
        assert CorrectionRules.load(Path("foo.yaml")).rules[0].name == "foo"

    def test_load_invalid_rule_raises(self, mocker: MockerFixture) -> None:
        mocker.patch.object(Path, "read_text", return_value="rules:\n  - {name: foo}\n")
        with pytest.raises(ValidationError):
            CorrectionRules.load(Path("foo.yaml"))

    @pytest.mark.parametrize(
        "side_effect, expected",
        [
            ([True, False], True),
            ([False, False], False),
        ],
    )
    def test_apply_rules_returns_whether_any_rule_acted(
        self, mocker: MockerFixture, side_effect: list[bool], expected: bool
    ) -> None:
        mock_apply: MagicMock = mocker.patch.object(
            CorrectionRule,
            "apply",
            autospec=True,
            side_effect=[({"foo": 1}, acted) for acted in side_effect],
        )
        rules: CorrectionRules = CorrectionRules(
            rules=[make_rule(), make_rule(name="qux")]
        )
        assert rules.apply({"foo": 1}, "quux") == ({"foo": 1}, expected)
        assert [call.args[1:] for call in mock_apply.call_args_list] == [
            ({"foo": 1}, "quux"),
            ({"foo": 1}, "quux"),
        ]
