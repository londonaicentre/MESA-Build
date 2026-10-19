import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from datagen.correction_rules import CorrectionRule
from datagen.types import JSONValue
from dry_run import Change, DryRun


class MockSchema(BaseModel):
    field: str


class DryRunFixture(DryRun):
    def apply(
        self, rules: list[CorrectionRule]
    ) -> tuple[dict[str, list[Change]], list[str]]:
        return self._apply(rules)

    @staticmethod
    def render(rule_name: str, change: Change, chars: int) -> str:
        return DryRun._render(rule_name, change, chars)


@dataclass
class ConstructorMocks:
    Corpus: MagicMock
    getLogger: MagicMock
    load: MagicMock


def make_rule(name: str) -> MagicMock:
    rule: MagicMock = MagicMock()
    rule.name = name
    return rule


@pytest.fixture
def constructor_mocks(mocker: MockerFixture) -> ConstructorMocks:
    corpus: MagicMock = mocker.patch("dry_run.Corpus")
    corpus.return_value.get_schema.return_value = MockSchema
    return ConstructorMocks(
        Corpus=corpus,
        getLogger=mocker.patch("dry_run.logging.getLogger"),
        load=mocker.patch(
            "dry_run.CorrectionRules.load",
            return_value=MagicMock(rules=[make_rule("foo"), make_rule("bar")]),
        ),
    )


@pytest.fixture
def dry_run(constructor_mocks: ConstructorMocks) -> DryRunFixture:
    return DryRunFixture("baz", Path("qux.yaml"))


class TestApply:
    @pytest.mark.parametrize(
        "second_pass, expected_unstable", [(False, []), (True, ["foo"])]
    )
    def test_apply_rules_returns_changes_and_unstable_records(
        self,
        dry_run: DryRunFixture,
        mocker: MockerFixture,
        constructor_mocks: ConstructorMocks,
        second_pass: bool,
        expected_unstable: list[str],
    ) -> None:
        constructor_mocks.Corpus.records.return_value = [
            ("", MagicMock(recordId="foo")),
            ("", MagicMock(recordId="bar")),
        ]
        constructor_mocks.Corpus.note.return_value = "quux"
        mocker.patch(
            "dry_run.extract_and_validate_json",
            side_effect=[(MockSchema(field="corge"), None), (None, None)],
        )
        after: dict[str, JSONValue] = {"field": "grault"}
        rule: MagicMock = make_rule("garply")
        rule.apply.side_effect = [(after, True), (after, second_pass)]
        changes, unstable = dry_run.apply([rule])
        assert changes == {"garply": [Change("foo", "quux", {"field": "corge"}, after)]}
        assert unstable == expected_unstable

    def test_apply_invalid_correction_raises(
        self,
        dry_run: DryRunFixture,
        mocker: MockerFixture,
        constructor_mocks: ConstructorMocks,
    ) -> None:
        constructor_mocks.Corpus.records.return_value = [
            ("", MagicMock(recordId="foo"))
        ]
        mocker.patch(
            "dry_run.extract_and_validate_json",
            return_value=(MockSchema(field="corge"), None),
        )
        rule: MagicMock = make_rule("garply")
        rule.apply.return_value = ({"field": None}, True)
        with pytest.raises(ValueError):
            dry_run.apply([rule])


class TestRender:
    def test_render_change_returns_diff_and_truncated_note(self) -> None:
        rendered: str = DryRunFixture.render(
            "foo", Change("bar", "abcdef", {"field": "baz"}, {"field": "qux"}), 3
        )
        assert rendered.startswith("=== foo: bar\n@@")
        assert '- "field": "baz"' in rendered
        assert '+ "field": "qux"' in rendered
        assert rendered.endswith("\nNOTE\nabc")


class TestRun:
    @pytest.mark.parametrize(
        "rule_name, expected_rules", [(None, ["foo", "bar"]), ("bar", ["bar"])]
    )
    def test_run_rule_name_applies_matching_rules(
        self,
        dry_run: DryRunFixture,
        mocker: MockerFixture,
        rule_name: str | None,
        expected_rules: list[str],
    ) -> None:
        apply: MagicMock = mocker.patch.object(DryRun, "_apply", return_value=({}, []))
        dry_run.run(rule_name, 0, 600)
        assert [rule.name for rule in apply.call_args.args[0]] == expected_rules

    def test_run_unknown_rule_raises(self, dry_run: DryRunFixture) -> None:
        with pytest.raises(ValueError):
            dry_run.run("quux", 0, 600)

    @pytest.mark.parametrize("show, expected_calls", [(0, 1), (1, 2)])
    def test_run_changes_reports_counts_and_shows_changes(
        self,
        dry_run: DryRunFixture,
        mocker: MockerFixture,
        constructor_mocks: ConstructorMocks,
        show: int,
        expected_calls: int,
    ) -> None:
        change: Change = Change("quux", "corge", {}, {})
        mocker.patch.object(
            DryRun, "_apply", return_value=({"foo": [change], "bar": []}, [])
        )
        render: MagicMock = mocker.patch.object(
            DryRun, "_render", return_value="grault"
        )
        dry_run.run(None, show, 600)
        info: MagicMock = constructor_mocks.getLogger.return_value.info
        assert info.call_args_list[0].args[0] == (
            "foo: 1\nbar: 0\nRecords changed: 1"
            "\nRecords changed again by a second pass: 0 []"
        )
        assert info.call_count == expected_calls
        if show:
            render.assert_called_once_with("foo", change, 600)
            assert info.call_args_list[1].args[0] == "grault"


class TestMain:
    def test_main_arguments_runs_dry_run(
        self,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        constructor_mocks: ConstructorMocks,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dry_run.py",
                "--schema",
                "baz",
                "qux.yaml",
                "--rule",
                "foo",
                "--show",
                "2",
            ],
        )
        run: MagicMock = mocker.patch.object(DryRun, "run")
        DryRun.main()
        constructor_mocks.load.assert_called_once_with(Path("qux.yaml"))
        run.assert_called_once_with("foo", 2, 600)
