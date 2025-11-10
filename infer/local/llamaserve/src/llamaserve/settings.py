from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class WeightSettings(BaseModel):
    ID: str = ''
    KEY: str = ''
    AWS_REGION: str = 'eu-west-2'


class ServerSettings(BaseModel):
    PORT: int = 5000
    PRECISION: str = 'float16'
    MAX_MODEL_LENGTH: int = 67015


class Settings(BaseSettings):
    MODEL: str = 'genollama1alpha01'
    WEIGHTS: WeightSettings = WeightSettings()
    SERVER: ServerSettings = ServerSettings()

    model_config = SettingsConfigDict(
        env_file='.env',
        env_nested_delimiter='_',
        extra='allow',
    )
