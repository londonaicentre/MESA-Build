import os
import json
from genomicextractmodel import GenomicTestReport
from pydantic import ValidationError

examples_folder = "datagen/examples/"

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
