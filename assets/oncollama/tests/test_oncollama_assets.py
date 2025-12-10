from typing import Any

import pytest

from oncollama_assets.wrapper import OncoLlamaAssets
from oncollama_assets.schema import OncoLlamaModel
from schemallama_types.assets.profile import Profile


@pytest.fixture(scope="session")
def oncollama_assets() -> OncoLlamaAssets:
    return OncoLlamaAssets()


def test_validate_schema(oncollama_assets: OncoLlamaAssets) -> None:
    result: bool
    message: str
    json_schema: dict[str, Any] | None
    result, message, json_schema = oncollama_assets.validate_schema(OncoLlamaModel)
    assert result
    assert message == "Schema validation successful"
    assert (
        json_schema
        and "document_has_primary_cancer_flag" in json_schema["properties"].keys()
    )


def test_load_system_prompt_datagen(oncollama_assets: OncoLlamaAssets) -> None:
    systemprompt_infer: str = oncollama_assets.load_system_prompt()
    # contains boilerplate text
    assert "CANCER CLINICAL DOCUMENT EXTRACTION" in systemprompt_infer
    # contains schema
    assert "class OncoLlamaModel(BaseModel)" in systemprompt_infer


def test_load_datagen_user_prompt(oncollama_assets: OncoLlamaAssets) -> None:
    assert "foo" in oncollama_assets.load_datagen_user_prompt({"content": "foo"})


def test_load_profiles_from_file(oncollama_assets: OncoLlamaAssets) -> None:
    profiles: list[Profile] = oncollama_assets.load_all_profiles()
    assert len(profiles) == 2550
    assert profiles[0].descriptive_name == "Cholangiocarcinoma"


def test_format_profile_prompt(oncollama_assets: OncoLlamaAssets) -> None:
    profiles: list[Profile] = oncollama_assets.load_all_profiles()
    profile_prompt_portion: str = oncollama_assets.format_profile_prompt(profiles[0])
    # contains boilerplate text
    assert "## USE THIS PRIMARY CANCER PROFILE" in profile_prompt_portion
    # contains an example
    assert "Cholangiocarcinoma" in profile_prompt_portion
