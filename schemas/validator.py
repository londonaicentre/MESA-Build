import os
import json
from geneworks_models import GenomicTestReport
from pydantic_core._pydantic_core import ValidationError

examples_folder = "../examples/"
problems = []

for example in os.listdir(examples_folder):
    filepath = os.path.join(examples_folder, example)

    with open(filepath, "r") as f:
        json_data = json.loads(f.read())
        try:
            loaded_example = GenomicTestReport(**json_data["output"])
            loaded_example.model_dump_json()
        except ValidationError:
            problems.append(filepath)
            print(f"Example {filepath} failed validation")
            loaded_example = GenomicTestReport(**json_data["output"])

print("All examples are validated.")
