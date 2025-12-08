from typing import Any

from datagen.claude import SampleGenerator
from litellm import BaseModel
import pandas as pd


class TestSampleGenerator(SampleGenerator):
    __test__ = False

    def extract_json_from_response(self, response: str) -> dict[str, Any] | None:
        return self._extract_json_from_response(response)

    def validate_with_pydantic(
        self, output_data: dict[str, Any]
    ) -> tuple[bool, BaseModel | None]:
        return self._validate_with_pydantic(output_data)

    def extract_validate_and_save_sample(self, response: str, sample_id: int) -> bool:
        return self._extract_validate_and_save_sample(response, sample_id)

    def generate_sample(self, bootstrap_file: pd.DataFrame, idx: int) -> bool:
        return self._generate_sample(bootstrap_file, idx)

    def process_bootstrap_rows(self, sample_size: int) -> tuple[int, int]:
        return self._process_bootstrap_rows(sample_size)
