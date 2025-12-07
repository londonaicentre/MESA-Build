from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BEDROCK_API_KEY: str = Field(default="", validation_alias="llm__anthropic__api_key")
    BEDROCK_EXECUTION_ROLE: str = ""
    BUCKET: str = ""
    US_BUCKET: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", extra="allow", populate_by_name=True
    )
