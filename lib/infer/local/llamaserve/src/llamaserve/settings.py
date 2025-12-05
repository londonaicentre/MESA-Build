from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class WeightSettings(BaseModel):
    ID: str = ''
    KEY: str = ''
    REGION: str = 'eu-west-2'


class ModelSettings(BaseModel):
    NAME: str = 'schemallama'
    PRECISION: str = 'float16'
    LENGTH: int = 41152


class ServerSettings(BaseModel):
    PORT: int = 5000


class Settings(BaseSettings):
    MODEL: ModelSettings = ModelSettings()
    WEIGHTS: WeightSettings = WeightSettings()
    SERVER: ServerSettings = ServerSettings()

    model_config = SettingsConfigDict(
        env_file='.env',
        env_nested_delimiter='_',
        extra='allow',
    )
