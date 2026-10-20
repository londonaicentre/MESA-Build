from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from utils.prompt import BasePromptBuilder


class FooSchema(BaseModel):
    foo: str


class FooParameters(BaseModel):
    condition: str
    example: str | None = None


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


class TestGetParameters:
    @pytest.mark.parametrize("parameters", [FooParameters, None])
    def test_get_parameters_constructed_returns_parameters(
        self, files: MagicMock, parameters: type[BaseModel] | None
    ) -> None:
        assert BasePromptBuilder("foo", FooSchema, parameters).get_parameters() is (
            parameters
        )


class TestSubstituteParameters:
    def test_substitute_parameters_values_given_replaces_placeholders(
        self, files: MagicMock
    ) -> None:
        assert (
            BasePromptBuilderFixture(
                "foo", FooSchema, FooParameters
            ).substitute_parameters(
                "{CONDITION} {EXAMPLE} {SCHEMA}",
                {"condition": "bar", "example": '{"foo": "baz"}'},
            )
            == 'bar {"foo": "baz"} {SCHEMA}'
        )

    def test_substitute_parameters_optional_value_absent_keeps_placeholder(
        self, files: MagicMock
    ) -> None:
        assert BasePromptBuilderFixture(
            "foo", FooSchema, FooParameters
        ).substitute_parameters("{CONDITION} {EXAMPLE}", {"condition": "bar"}) == (
            "bar {EXAMPLE}"
        )

    @pytest.mark.parametrize("parameters", [None, {}])
    def test_substitute_parameters_no_model_no_values_returns_prompt(
        self, files: MagicMock, parameters: dict[str, str] | None
    ) -> None:
        assert (
            BasePromptBuilderFixture("foo", FooSchema).substitute_parameters(
                "{CONDITION}", parameters
            )
            == "{CONDITION}"
        )

    def test_substitute_parameters_no_model_values_given_raises(
        self, files: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="does not take prompt parameters"):
            BasePromptBuilderFixture("foo", FooSchema).substitute_parameters(
                "{CONDITION}", {"condition": "bar"}
            )

    def test_substitute_parameters_unknown_value_raises(self, files: MagicMock) -> None:
        with pytest.raises(ValueError, match=r"Unknown prompt parameters: \['baz'\]"):
            BasePromptBuilderFixture(
                "foo", FooSchema, FooParameters
            ).substitute_parameters("{CONDITION}", {"condition": "bar", "baz": "qux"})

    @pytest.mark.parametrize("parameters", [None, {"example": "bar"}])
    def test_substitute_parameters_missing_value_raises(
        self, files: MagicMock, parameters: dict[str, str] | None
    ) -> None:
        with pytest.raises(ValidationError):
            BasePromptBuilderFixture(
                "foo", FooSchema, FooParameters
            ).substitute_parameters("{CONDITION}", parameters)


class TestBuildDatagenPrompt:
    def test_build_datagen_prompt_parameters_given_substitutes_then_fills_assets(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        assert BasePromptBuilder("foo", FooSchema, FooParameters).build_datagen_prompt(
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
            BasePromptBuilder("foo", FooSchema, FooParameters).build_main_prompt(
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
