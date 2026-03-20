import importlib.metadata
from datetime import date
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field


class ModelCard(BaseModel):
    """Model card for MESA models"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_model_hf: str
    model_name: str
    major: int
    minor: int
    patch: int

    @computed_field  # type: ignore
    @property
    def model_version(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    model_description: str
    model_train_date: date = Field(default_factory=date.today)
    training_data: List[str]
    output_schema: type[BaseModel]

    @computed_field  # type: ignore
    @property
    def schema_name(self) -> str:
        module_name: str = self.output_schema.__module__.split(".")[0]
        return next(
            iter(importlib.metadata.packages_distributions().get(module_name, [])),
            module_name,
        )

    @computed_field  # type: ignore
    @property
    def schema_version(self) -> str:
        return importlib.metadata.version(self.schema_name)

    def __get_yaml_str(self) -> str:
        return yaml.dump(
            self.model_dump(exclude={"major", "minor", "patch", "output_schema"}),
            default_flow_style=False,
            sort_keys=False,
        )

    def to_yaml(self, path: str = "model_card.yml") -> None:
        with Path(path).open("w") as file:
            file.write(self.__get_yaml_str())

    def to_yaml_bytes(self) -> bytes:
        return self.__get_yaml_str().encode("utf-8")
