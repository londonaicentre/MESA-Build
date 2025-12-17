import json
from json import JSONDecodeError
from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from sagemaker.jumpstart.estimator import JumpStartEstimator

from finetune.config import Config
from utils.aws import AWS


class FineTuner:
    """Fine-tune a Llama model on AWS Sagemaker

    Args:
        instance_type (str): AWS instance type to use for fine-tuning

    """

    def __init__(self, instance_type: str):
        self.__logger: logging.Logger = logging.getLogger(__name__)
        self.__config: Config = Config()
        self.__instance_type: str = instance_type
        self.__model_id: str = self.__config.models["llama"].model
        self.__model_region: str = self.__config.models["llama"].region
        self.__model_version: str = self.__config.models["llama"].version
        self.__model_template_filename: str = self.__config.models[
            "llama"
        ].template_filename
        self.__model_train_filename: str = self.__config.models["llama"].train_filename

    def _generate_template_file(self, system_prompt: str) -> None:
        """Generate template file into which part of the training
            data are embedded.

        Args:
            system_prompt (str): Prompt for the head of this file

        """
        with open("template.json", "w") as outfile:
            print(
                json.dumps(
                    {
                        "prompt": system_prompt
                        + "\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n",
                        "completion": "{response}",
                    }
                ),
                file=outfile,
            )

    def _generate_train_file(self, samples_input_folder: str) -> bool:
        """Generate a training file formatted from sample data

        Args:
            samples_input_folder (str): path to sample data file

        """
        with open("train.jsonl", "w") as outfile:
            for file in Path(samples_input_folder).glob("*.json"):
                with open(file) as infile:
                    try:
                        parsed_text: dict[str, Any] = json.loads(infile.read())
                        print(
                            json.dumps(
                                {
                                    "instruction": "Extract the given information into a structured schema.",
                                    "context": parsed_text["content"],
                                    "response": json.dumps(
                                        parsed_text["output"], ensure_ascii=True
                                    ),
                                }
                            ),
                            file=outfile,
                        )
                    except JSONDecodeError as e:
                        self.__logger.error(e)
                        return False
        return True

    def __run_estimator(
        self,
        role: str,
        input_path: str,
        output_path: str,
        instruction_tuned: bool = True,
        quantized: bool = False,
    ) -> None:
        """Start fine-tuning process

        Args:
            role (str): The ARN of an IAM role with permissions to
                access SageMaker
            input_path (str): Path in S3 to the training data
            output_path (str): Path in S3 to place fine-tuned
                model weights
            instruction_tuned (bool, optional): Whether to instruction-train
                the model (indicated by `template.json`). Defaults to True.
            quantized (bool, optional): Whether to load the base model in
                lower precision. Defaults to True.

        """
        estimator: JumpStartEstimator = JumpStartEstimator(
            model_id=self.__model_id,
            model_version=self.__model_version,
            role=role,
            disable_output_compression=True,
            output_path=output_path,
            instance_type=self.__instance_type,
            region=self.__model_region,
            environment={"accept_eula": "true"},
        )
        estimator.set_hyperparameters(
            instruction_tuned="True" if instruction_tuned else "False",
            chat_dataset="False" if instruction_tuned else "True",
            int8_quantization="True" if quantized else "False",
            enable_fsdp="False" if quantized else "True",
        )  # type: ignore[no-untyped-call]
        estimator.fit({"training": input_path})

    def run_finetune(
        self,
        system_prompt: str,
        sample_data_file: str,
        bucket: str,
        sagemaker_execution_role: str,
        dry_run: bool = False,
    ) -> None:
        """Start fine-tuning pipeline

        Args:
            system_prompt (str): Prompt for fine-tuning template
            sample_data_file (str): File containing sample data to
                use in fine-tuning process
            bucket (str): S3 bucket where fine-tuning input files
                and fine-tuned weights are/will be stored
            sagemaker_execution_role (str): The ARN of an IAM role
                with permissions to access SageMaker
            dry_run (bool, optional): Whether to run all processes
                except calls to AWS. Default to False.

        """
        job_id: str = "finetune/" + datetime.now().strftime("%Y-%m-%d-%H%M")
        self._generate_template_file(system_prompt)
        upload_path: str = job_id + "/input"
        if not dry_run:
            AWS.upload_file(
                self.__model_region,
                self.__model_template_filename,
                bucket,
                self.__model_template_filename,
                upload_path,
            )
        self._generate_train_file(sample_data_file)
        if not dry_run:
            AWS.upload_file(
                self.__model_region,
                self.__model_train_filename,
                bucket,
                self.__model_train_filename,
                upload_path,
            )
        if not dry_run:
            self.__run_estimator(
                sagemaker_execution_role,
                "s3://" + bucket + "/" + job_id + "/input",
                "s3://" + bucket + "/" + job_id + "/output",
            )
