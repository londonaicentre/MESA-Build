from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from utils.prompt import BasePromptBuilder


class FooSchema(BaseModel):
    foo: str


class BasePromptBuilderFixture(BasePromptBuilder):
    def substitute_parameters(
        self, prompt: str, parameters: dict[str, str] | None
    ) -> str:
        return self._substitute_parameters(prompt, parameters)


@dataclass
class PromptBuilderMocks:
    files: MagicMock
    getmodule: MagicMock
    getsource: MagicMock
    _load_root: MagicMock
    _load: MagicMock
    _substitute_parameters: MagicMock


@pytest.fixture
def prompt_builder_mocks(mocker: MockerFixture) -> PromptBuilderMocks:
    return PromptBuilderMocks(
        files=mocker.patch("utils.prompt.files"),
        getmodule=mocker.patch("utils.prompt.inspect.getmodule"),
        getsource=mocker.patch(
            "utils.prompt.inspect.getsource",
            return_value='class FooSchema:\n    """{"bar": {"baz": 1}}"""',
        ),
        _load_root=mocker.patch.object(BasePromptBuilder, "_load_root"),
        _load=mocker.patch.object(
            BasePromptBuilder, "_load", return_value='{"foo": "qux"}'
        ),
        _substitute_parameters=mocker.patch.object(
            BasePromptBuilder,
            "_substitute_parameters",
            return_value="bar {SCHEMA} {EXAMPLE}",
        ),
    )


@pytest.fixture
def files(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("utils.prompt.files")


class TestInit:
    @pytest.mark.parametrize("name", ["schema", "example", "fooBar", "1foo", "foo-bar"])
    def test_init_invalid_parameter_name_raises(
        self, files: MagicMock, name: str
    ) -> None:
        with pytest.raises(ValueError, match="Invalid prompt parameter names"):
            BasePromptBuilder("foo", FooSchema, ("condition", name))


class TestGetParameters:
    @pytest.mark.parametrize("parameters", [("condition", "qualifier"), ()])
    def test_get_parameters_constructed_returns_parameters(
        self, files: MagicMock, parameters: tuple[str, ...]
    ) -> None:
        assert BasePromptBuilder("foo", FooSchema, parameters).get_parameters() == (
            parameters
        )


class TestSubstituteParameters:
    def test_substitute_parameters_values_given_replaces_placeholders(
        self, files: MagicMock
    ) -> None:
        assert (
            BasePromptBuilderFixture(
                "foo", FooSchema, ("condition", "qualifier")
            ).substitute_parameters(
                "{CONDITION} {QUALIFIER} {CONDITION} {SCHEMA}",
                {"condition": "bar", "qualifier": "baz"},
            )
            == "bar baz bar {SCHEMA}"
        )

    @pytest.mark.parametrize(
        "value", ["Crohn's disease", "type 2 diabetes (T2DM)", "Ménière's, left-sided."]
    )
    def test_substitute_parameters_allowed_value_replaces_placeholder(
        self, files: MagicMock, value: str
    ) -> None:
        assert (
            BasePromptBuilderFixture(
                "foo", FooSchema, ("condition",)
            ).substitute_parameters("{CONDITION}", {"condition": value})
            == value
        )

    @pytest.mark.parametrize("parameters", [None, {}])
    def test_substitute_parameters_no_parameters_returns_prompt(
        self, files: MagicMock, parameters: dict[str, str] | None
    ) -> None:
        assert (
            BasePromptBuilderFixture("foo", FooSchema).substitute_parameters(
                "{CONDITION}", parameters
            )
            == "{CONDITION}"
        )

    @pytest.mark.parametrize(
        "value", ["{SCHEMA}", "foo\nbar", "foo\n", "foo: bar", "", "f" * 101]
    )
    def test_substitute_parameters_disallowed_value_raises(
        self, files: MagicMock, value: str
    ) -> None:
        with pytest.raises(ValidationError):
            BasePromptBuilderFixture(
                "foo", FooSchema, ("condition",)
            ).substitute_parameters("{CONDITION}", {"condition": value})

    @pytest.mark.parametrize(
        ("parameters", "values"),
        [
            ((), {"condition": "bar"}),
            (("condition",), {"condition": "bar", "baz": "qux"}),
            (("condition",), None),
            (("condition", "qualifier"), {"condition": "bar"}),
        ],
    )
    def test_substitute_parameters_unknown_or_missing_key_raises(
        self,
        files: MagicMock,
        parameters: tuple[str, ...],
        values: dict[str, str] | None,
    ) -> None:
        with pytest.raises(ValidationError):
            BasePromptBuilderFixture(
                "foo", FooSchema, parameters
            ).substitute_parameters("{CONDITION}", values)


class TestBuildDatagenPrompt:
    def test_build_datagen_prompt_parameters_given_substitutes_then_fills_assets(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        assert BasePromptBuilder("foo", FooSchema, ("condition",)).build_datagen_prompt(
            {"condition": "bar"}
        ) == ('bar class FooSchema:\n    """{"bar": {"baz": 1}}""" {"foo": "qux"}')
        prompt_builder_mocks._substitute_parameters.assert_called_once_with(
            prompt_builder_mocks._load_root.return_value, {"condition": "bar"}
        )
        prompt_builder_mocks._load_root.assert_called_once_with("prompt_datagen.txt")

    def test_build_datagen_prompt_module_not_found_raises(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        prompt_builder_mocks.getmodule.return_value = None

        with pytest.raises(ValueError, match="module not found"):
            BasePromptBuilder("foo", FooSchema).build_datagen_prompt()


class TestBuildMainPrompt:
    def test_build_main_prompt_parameters_given_substitutes_then_fills_schema(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        assert (
            BasePromptBuilder("foo", FooSchema, ("condition",)).build_main_prompt(
                {"condition": "bar"}
            )
            == 'bar class FooSchema:\n    """{"bar": {"baz": 1}}""" {EXAMPLE}'
        )
        prompt_builder_mocks._substitute_parameters.assert_called_once_with(
            prompt_builder_mocks._load_root.return_value, {"condition": "bar"}
        )
        prompt_builder_mocks._load_root.assert_called_once_with("prompt_main.txt")

    def test_build_main_prompt_module_not_found_raises(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        prompt_builder_mocks.getmodule.return_value = None

        with pytest.raises(ValueError, match="module not found"):
            BasePromptBuilder("foo", FooSchema).build_main_prompt()
