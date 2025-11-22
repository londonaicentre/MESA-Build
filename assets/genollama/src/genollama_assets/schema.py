import os
import json
from genollama_assets.genollama_assets_types import GenomicTestReport
from pydantic import ValidationError

def tojson():
    schema = GenomicTestReport.model_json_schema()

    version = "03"

    with open(f"schema_v{version}.json", "w") as f:
        json.dump(schema, f, indent=4)

    print("Schema generated")

def validator():
    examples_folder = "examples/"

    for example in os.listdir(examples_folder):
        filepath = os.path.join(examples_folder, example)

        with open(filepath, "r") as f:
            json_data = json.loads(f.read())
            try:
                loaded_example = GenomicTestReport(**json_data["output"])
                print(f"Example {filepath} validated")
                loaded_example.model_dump_json()
            except ValidationError:
                print(f"Example {filepath} failed validation")
                loaded_example = GenomicTestReport(**json_data["output"])

    print("All examples checked.")
