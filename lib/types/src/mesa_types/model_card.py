import importlib.metadata
from datetime import date
from pathlib import Path
from typing import Any, List

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


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
    output_schema: type[BaseModel] | None = None
    schema_name: str | None = None
    schema_version: str | None = None

    @model_validator(mode="after")
    def _derive_schema_info(self) -> "ModelCard":
        if self.output_schema is not None and self.schema_name is None:
            module_name: str = self.output_schema.__module__.split(".")[0]
            self.schema_name = next(
                iter(importlib.metadata.packages_distributions().get(module_name, [])),
                module_name,
            )

        if self.schema_name is not None and self.schema_version is None:
            self.schema_version = importlib.metadata.version(self.schema_name)
        if self.schema_name is None and self.schema_version is not None:
            raise ValueError("schema_version requires schema_name to also be set")
        return self

    @model_validator(mode="before")
    @classmethod
    def _split_model_version(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and "model_version" in data
            and not {"major", "minor", "patch"} & data.keys()
        ):
            data["major"], data["minor"], data["patch"] = (
                int(part) for part in str(data["model_version"]).split(".")
            )
        return data

    @computed_field  # type: ignore
    @property
    def model_identifier(self) -> str:
        return f"{self.model_name}_{self.major}_{self.minor}_{self.patch}"

    def __get_yaml_str(self) -> str:
        return yaml.dump(
            self.model_dump(
                exclude={"major", "minor", "patch", "output_schema", "model_identifier"}
            ),
            default_flow_style=False,
            sort_keys=False,
        )

    def to_yaml(self, path: str = "model_card.yml") -> None:
        with Path(path).open("w") as file:
            file.write(self.__get_yaml_str())

    def to_yaml_bytes(self) -> bytes:
        return self.__get_yaml_str().encode("utf-8")
