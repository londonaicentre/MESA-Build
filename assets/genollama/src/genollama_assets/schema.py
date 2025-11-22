import os
import json
from typing import Any

from pydantic import ValidationError

from genollama_assets.genollama_assets_types import GenomicTestReport

def tojson() -> None:
    schema: dict[str, Any] = GenomicTestReport.model_json_schema()

    version: str = "03"

    with open(f"schema_v{version}.json", "w") as f:
        json.dump(schema, f, indent=4)

    print("Schema generated")

def validator() -> None:
    examples_folder: str = "examples/"

    for example in os.listdir(examples_folder):
        filepath: str = os.path.join(examples_folder, example)

        with open(filepath, "r") as f:
            json_data: dict[str, Any] = json.loads(f.read())
            try:
                loaded_example: GenomicTestReport = GenomicTestReport(**json_data["output"])
                print(f"Example {filepath} validated")
                loaded_example.model_dump_json()
            except ValidationError:
                print(f"Example {filepath} failed validation")
                loaded_example = GenomicTestReport(**json_data["output"])

    print("All examples checked.")
