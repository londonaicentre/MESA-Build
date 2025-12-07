from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BEDROCK_EXECUTION_ROLE: str = ""
    BUCKET: str = ""
    US_BUCKET: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
    )
