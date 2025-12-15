import pytest

from oncollama_assets.wrapper import OncoLlamaAssets
from schemallama_types.assets.profile import Profile


@pytest.fixture(scope="session")
def oncollama_assets() -> OncoLlamaAssets:
    return OncoLlamaAssets()


def test_validate_schema(oncollama_assets: OncoLlamaAssets) -> None:
    # Validate that we can instantiate and validate the schema
    # This would probably fail at an earlier stage if there were issues in reality.
    oncollama_assets.schema.model_json_schema()


def test_load_system_prompt_datagen(oncollama_assets: OncoLlamaAssets) -> None:
    systemprompt_infer: str = oncollama_assets.load_system_prompt()
    # contains boilerplate text
    assert "CANCER CLINICAL DOCUMENT EXTRACTION" in systemprompt_infer
    # doesn't contain python schema
    assert "class OncoLlamaModel(BaseModel)" not in systemprompt_infer
    # contains json schema
    assert "#/$defs/PerformanceStatus" in systemprompt_infer


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
