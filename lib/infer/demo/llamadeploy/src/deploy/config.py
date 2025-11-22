import json
from importlib.resources import files

from pydantic_settings import BaseSettings
from pydantic import BaseModel

class ImageConfig(BaseModel):
    version: str
    pytorch_version: str
    py_version: str

class ModelConfig(BaseModel):
    endpoint_name: str
    hf_model_name: str
    region: str

class Config(BaseSettings):
    models: dict[str, ModelConfig]
    images: dict[str, ImageConfig]

    def __init__(self) -> None:
        super().__init__(
            models=json.loads(files("deploy").joinpath("config/models.json").read_text()),
            images=json.loads(files("deploy").joinpath("config/images.json").read_text())
        )