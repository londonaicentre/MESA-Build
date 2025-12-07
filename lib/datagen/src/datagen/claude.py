import logging
import os
import re
from re import Match
import json
from datetime import datetime
from typing import Any, Callable, cast

import pandas as pd
import litellm
from pydantic import BaseModel
from litellm import Choices, ModelResponse

from datagen.config import Config
from utils.aws import AWS
from utils.llm import BatchOutputs

litellm.suppress_debug_info = (
    True  # suppress unhelpful library output on rate limit error
)


class SampleGenerator:
    """Claude synthetic data generator for fine-tuning
        schema standardisation LLMs.

    Args:
        system_prompt (str): System prompt for sample generation
        user_prompt_function (Callable): User prompt generation
            function to which bootstrap data is passed
        schema (type[BaseModel]): Schema class to use for validation
        model_name (str): Name of model to use on AWS Bedrock
        bootstrap_file_path (str): Path to bootstrap (template) file
        bedrock_api_key (str, optional): API key to access AWS Bedrock.
            Defaults to None.

    """

    def __init__(
        self,
        system_prompt: str,
        user_prompt_function: Callable[[dict[str, Any]], str],
        schema: type[BaseModel],
        model_name: str,
        bootstrap_file_path: str,
        bedrock_api_key: str | None = None,
    ):
        self.__config: Config = Config()
        self.__logger: logging.Logger = logging.getLogger(__name__)
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())
        self.__system_prompt: str = system_prompt
        self.__user_prompt_function: Callable[[dict[str, Any]], str] = (
            user_prompt_function
        )
        self.__schema: type[BaseModel] = schema
        self.__model_id: str = self.__config.models[model_name].model
        self.__model_region: str = self.__config.models[model_name].region
        os.environ["AWS_REGION_NAME"] = self.__model_region
        self.__model_batch_file: str = self.__config.models[model_name].batch_file
        self.__bootstrap_file_path: str = bootstrap_file_path
        self.__bedrock_api_key: str | None = bedrock_api_key
        self.__output_folder_name: str = f"samples_{model_name}/"

    def __extract_json_from_response(self, response: str) -> dict[str, Any] | None:
        """Extract JSON from Claude's response

        Args:
            response (str): Claude's response

        Returns:
            JSON if parse successful, None otherwise

        """

        # try to parse response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # try to find JSON in a code block
            json_match: Match[str] | None = re.search(
                r"```json\s*(.*?)\s*```", response, re.DOTALL
            )
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # If fails, try to find any JSON-like structure
            json_match = re.search(r"{[\s\S]*}", response)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    raise Exception("Could not parse JSON from response")
            self.__logger.error("No valid JSON found in response")
            return None

    def __validate_with_pydantic(
        self, output_data: dict[str, Any]
    ) -> tuple[bool, BaseModel | None]:
        """Validate the output against the Pydantic schema

        Args:
            output_data (dict): JSON data to validate

        Returns:
            True, Parsed JSON if validation successful, False, None otherwise.

        """
        try:
            validated_report: BaseModel = self.__schema(**output_data)
            return True, validated_report
        except Exception as e:
            self.__logger.error(f"Pydantic validation error: {e}")
            return False, None

    def __extract_validate_and_save_sample(self, response: str, sample_id: int) -> bool:
        """Extract sample from model response, validate it and save it to a
            labelled file.

        Args:
            response (str): The raw response
            sample_id (int): The id with which to label the sample

        Returns:
            bool: Whether the named operations were successful

        """
        json_output: dict[str, Any] | None = self.__extract_json_from_response(response)
        if json_output is not None:
            if (
                not isinstance(json_output, dict)
                or "content" not in json_output
                or "output" not in json_output
            ):
                self.__logger.error(
                    f"Invalid schema format in output for sample {sample_id + 1}"
                )
                return False

            # validate against schema
            is_valid: bool
            validated_output: BaseModel | None
            is_valid, validated_output = self.__validate_with_pydantic(
                json_output["output"]
            )
            if is_valid and validated_output is not None:
                # Convert Pydantic model to dict for JSON serialization
                json_output["output"] = validated_output.model_dump()
                output_filename: str = os.path.join(
                    self.__output_folder_name, f"sample{sample_id + 1:04d}.json"
                )
                try:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        json.dump(json_output, f, indent=4, ensure_ascii=False)
                    self.__logger.info(
                        f"Successfully saved output to {output_filename}"
                    )
                    return True
                except Exception as e:
                    self.__logger.error(f"Error saving JSON to file: {e}")
                    return False
            else:
                self.__logger.error(
                    f"Pydantic validation failed for sample {sample_id + 1}"
                )

                # for debugging later
                debug_filename: str = os.path.join(
                    self.__output_folder_name,
                    f"invalid_sample{sample_id + 1:04d}.json",
                )
                with open(debug_filename, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)
                return False
        else:
            self.__logger.warning(
                f"Skipping file save for sample {sample_id + 1} due to JSON parsing failure"
            )
            self.__logger.debug(
                "Claude response:",
                response,
            )
            return False

    def __generate_sample(self, bootstrap_file: pd.DataFrame, idx: int) -> bool:
        """Generate structured output from a synthetic patient report.

        Args:
            bootstrap_file (pandas.DataFrame): Specialised examples (template) for
                sample report generation
            idx (int): Index for row to be processed from bootstrap file

        Returns:
            bool: Whether sample generation is successful

        """
        row: pd.Series = bootstrap_file.iloc[idx]
        try:
            user_prompt: str = self.__user_prompt_function(row.to_dict())
            if self.__bedrock_api_key is None:
                return False
            message: ModelResponse | None = AWS.bedrock_completion(
                self.__model_id,
                self.__system_prompt,
                user_prompt,
                self.__bedrock_api_key,
            )
            if message is None:
                return False
            content: str | None = cast(Choices, message.choices[0]).message.content
            if content is not None:
                return self.__extract_validate_and_save_sample(content, idx)
            else:
                return False
        except Exception as e:
            self.__logger.error(f"Error processing row {idx + 1}: {e}")
            return False

    # real-time generation

    def __process_bootstrap_rows(
        self,
        sample_size: int = 10,
    ) -> None:
        """Process rows from the specified bootstrap file and
            generate the requested number of samples

        Args:
            bedrock_api_key (str): API key to access AWS Bedrock
            sample_size (int, optional): Number of samples to generate.
                Defaults to 10.

        """
        bootstrap_file: pd.DataFrame = pd.read_csv(self.__bootstrap_file_path)

        # Process rows
        os.makedirs(self.__output_folder_name, exist_ok=True)
        samples_exist: int = len(os.listdir(self.__output_folder_name))
        successful_generations: int = 0
        failed_generations: int = 0
        if samples_exist > sample_size:
            self.__logger.warning(
                f"Requested number of samples have already been generated in {self.__output_folder_name}."
            )
            exit()
        max_samples: int = len(bootstrap_file.index)
        if sample_size > max_samples:
            self.__logger.warning(
                f"Requested number of samples is more than number of templates for generation. \
                    Will create {max_samples} samples instead of {sample_size}"
            )
            sample_size = max_samples
        for idx, row in bootstrap_file.iterrows():
            id: int = int(cast(int, idx))

            # Skip rows for which samples have been generated
            if id < samples_exist:
                continue

            # Stop generating samples when requested amount is reached
            if id == sample_size:
                self.__logger.info(
                    f"Generated the requested number of samples, {sample_size}."
                )
                break
            if self.__generate_sample(bootstrap_file, id):
                successful_generations += 1
            else:
                failed_generations += 1
        self.__logger.info(
            f"Processing complete: {successful_generations} successful, {failed_generations} failed"
        )

    def run_sample_generation(self, sample_size: int) -> None:
        """Generate samples via individual AWS Bedrock inference calls

        Args:
            sample_size (int): Number of samples to be generated

        """

        # Generate samples from bootstrap file
        self.__process_bootstrap_rows(
            sample_size,
        )

    # backfill

    def __find_missing_idx(self, sample_size: int) -> list[int]:
        """For a given folder and expected number of samples,
            identifies indices for which no sample was generated

        Args:
            sample_size (int): Expected number of samples

        Returns:
            list[int]: List of indices without a sample generated

        """
        all_files: list[str] = os.listdir(self.__output_folder_name)
        filenames: list[str] = [
            file.strip(".json").strip("sample") for file in all_files
        ]
        missing_idx: list[int] = []
        for idx in range(sample_size):
            if f"{idx + 1:04d}" not in filenames:
                missing_idx.append(idx)
        return missing_idx

    def __backfill(self, idx_list: list[int]) -> None:
        """Generate samples for the missing indices

        Args:
            idx_list (list[int]): List of indices for a sample to be generated

        """
        bootstrap_file: pd.DataFrame = pd.read_csv(self.__bootstrap_file_path)
        successful_generations: int = 0
        failed_generations: int = 0
        for idx in idx_list:
            self.__logger.debug(f"Processing row {idx + 1}")
            if self.__generate_sample(bootstrap_file, idx):
                successful_generations += 1
            else:
                failed_generations += 1
        self.__logger.info(
            f"Processing complete: {successful_generations} successful, {failed_generations} failed"
        )

    def run_backfill(self, sample_size: int) -> None:
        """Backfill missing samples

        Args:
            sample_size (int): Number of samples to be generated

        """

        # Generate samples for missed indices in the bootstrap file specified
        missing_idx: list[int] = self.__find_missing_idx(sample_size)
        self.__logger.info(f"There are {len(missing_idx)} samples missing")
        self.__backfill(missing_idx)

    # batch

    def __generate_batch(
        self, sample_size: int, file_name: str = "anthropic_batch_job.jsonl"
    ) -> str:
        """Generate batch request file for Anthropic model

        Args:
            sample_size (int): Number of samples to be generated

        Returns:
            str: The batch request file

        """
        bootstrap_file: pd.DataFrame = pd.read_csv(self.__bootstrap_file_path)
        max_samples: int = len(bootstrap_file.index)
        if sample_size > max_samples:
            self.__logger.warning(
                f"Requested number of samples is more than number of templates for generation. \
                    Will create {max_samples} samples instead of {sample_size}"
            )
            sample_size = max_samples
        with open(file_name, "w") as outfile:
            for idx, row in bootstrap_file.iterrows():
                # Stop generating samples when requested amount is reached
                if idx == sample_size:
                    self.__logger.debug(
                        f"Generated the requested number of samples, {sample_size}."
                    )
                    break
                print(
                    json.dumps(
                        AWS.create_anthropic_bedrock_batch_entry(
                            str(idx),
                            self.__system_prompt,
                            self.__user_prompt_function(row.to_dict()),
                        )
                    ),
                    file=outfile,
                )
        return file_name

    def run_batch_inference(
        self,
        sample_size: int,
        bucket: str,
        bedrock_execution_role: str,
    ) -> None:
        """Generate samples via batch inference

        Args:
            sample_size (int): Number of samples to be generated
            bucket (str): The name of the bucket to which the batch
                specification should be uploaded
            bedrock_execution_role (str): The ARN of an IAM role with
                permissions to access S3 for batch specification and
                access cross-region models

        """

        # Process all samples from bootstrap file in batch mode
        # Create batch instruction JSONL file
        self.__generate_batch(sample_size)
        AWS.run_batch_inference(
            "datagen/" + datetime.now().strftime("%Y-%m-%d-%H%M"),
            self.__model_id,
            self.__model_batch_file,
            bucket,
            bedrock_execution_role,
            self.__model_region,
        )

    def extract_batch_output(
        self, file_name: str = "anthropic_batch_job.jsonl"
    ) -> None:
        successful_generations: int = 0
        failed_generations: int = 0
        with open(file_name + ".out") as batch_output_file:
            for sample_id, bedrock_batch_output in enumerate(
                BatchOutputs.model_validate(
                    {"outputs": [json.loads(line) for line in batch_output_file]}
                ).outputs
            ):
                try:
                    if self.__extract_validate_and_save_sample(
                        str(bedrock_batch_output.modelOutput.content[0].text), sample_id
                    ):
                        successful_generations += 1
                    else:
                        failed_generations += 1
                except Exception as e:
                    failed_generations += 1
                    self.__logger.error(f"Error processing row {sample_id + 1}: {e}")
        self.__logger.info(
            f"Processing complete: {successful_generations} successful, {failed_generations} failed"
        )


class BootstrapFileGenerator:
    @staticmethod
    def run_bootstrap_file_generation(
        system_prompt: str,
        user_prompt_function: Callable[[str], str],
        instruction: str,
        model_name: str,
        bedrock_api_key: str,
        bucket: str = "",
    ) -> None:
        """Generate bootstrap file to vary samples

        Args:
            system_prompt (str): System prompt for bootstrap file generation
            user_prompt_function (Callable): User prompt generation
                function to which instruction is passed
            instruction (str): Instruction to tailor bootstrap file to
                specific area
            model_name (str): Name of model to use on AWS Bedrock
            bedrock_api_key (str): API key to access AWS Bedrock
            bucket (str, optional): The name of the bucket in which to store the bootstrap file.
                If omitted, file is not backed up.

        """
        config: Config = Config()
        message: ModelResponse | None = AWS.bedrock_completion(
            config.models[model_name].model,
            system_prompt,
            user_prompt_function(instruction),
            bedrock_api_key,
        )
        bootstrap_file_name: str = "bootstrap.csv"
        if message is not None:
            with open(bootstrap_file_name, "w", newline="") as file:
                file.write(str(cast(Choices, message.choices[0]).message.content))
        if bucket != "":
            AWS.upload_file(
                config.models[model_name].region,
                bootstrap_file_name,
                bucket,
                bootstrap_file_name,
                "datagen/" + datetime.now().strftime("%Y-%m-%d-%H%M"),
            )
