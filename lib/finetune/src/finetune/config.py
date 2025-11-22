import json
from importlib.resources import files

from pydantic_settings import BaseSettings
from pydantic import BaseModel

class ModelConfig(BaseModel):
    model: str
    version: str
    region: str

class Config(BaseSettings):
    models: dict[str, ModelConfig]
    
    def __init__(self):
        super().__init__(models=json.loads(files("finetune").joinpath("config/config.json").read_text()))