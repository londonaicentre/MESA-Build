import pytest

from oncollama_assets.wrapper import OncoLlamaAssets
from oncollama_assets.schema import OncoLlamaModel


@pytest.fixture(scope="session")
def oncollama_assets() -> OncoLlamaAssets:
    return OncoLlamaAssets()


def test_validate_schema() -> None:
    # Validate that we can instantiate and validate the schema
    # This would probably fail at an earlier stage if there were issues in reality.
    OncoLlamaModel.model_json_schema()


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
