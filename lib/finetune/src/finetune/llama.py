import json
from datetime import datetime
from typing import Any

from sagemaker.jumpstart.estimator import JumpStartEstimator

from finetune.config import Config
from utils.aws import upload_file


def generate_template_file(system_prompt: str) -> None:
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


def generate_train_file(samples_input_file: str) -> None:
    """Generate a training file formatted from sample data

    Args:
        samples_input_file (str): path to sample data file

    """
    with open("train.jsonl", "w") as outfile:
        with open(samples_input_file, "r") as infile:
            for line in infile:
                parsed_line: dict[str, Any] = json.loads(line)
                text: str = (
                    parsed_line["modelOutput"]["content"][0]["text"]
                    .replace("```json", "")
                    .replace("```", "")
                )
                try:
                    parsed_text: dict[str, Any] = json.loads(text)
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
                except Exception:
                    pass


def run_estimator(
    model_id: str,
    model_version: str,
    role: str,
    input_path: str,
    output_path: str,
    instance_type: str,
    region: str,
    instruction_tuned: bool = True,
    quantized: bool = True,
) -> None:
    """Start fine-tuning process

    Args:
        model_id (str): The id of the model to fine-tune
        model_version (str): The version of the model to fine-tune
        role (str): The ARN of an IAM role with permissions to
            access SageMaker
        input_path (str): Path in S3 to the training data
        output_path (str): Path in S3 to place fine-tuned
            model weights
        instance_type (str): AWS instance type to use for fine-tuning
        region (str): Region in which to do the fine-tuning
        instruction_tuned (bool, optional): Whether to instruction-train
            the model (indicated by `template.json`). Defaults to True.
        quantized (bool, optional): Whether to load the base model in
            lower precision. Defaults to True.

    """
    estimator: JumpStartEstimator = JumpStartEstimator(
        model_id=model_id,
        model_version=model_version,
        role=role,
        disable_output_compression=True,
        output_path=output_path,
        instance_type=instance_type,
        region=region,
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
    system_prompt: str,
    sample_data_file: str,
    bucket: str,
    sagemaker_execution_role: str,
    instance_type: str,
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
        instance_type (str): AWS instance type to use for fine-tuning
        dry_run (bool, optional): Whether to run all processes
            except calls to AWS. Default to False.

    """
    config: Config = Config()
    job_id: str = "finetune/" + datetime.now().strftime("%Y-%m-%d-%H%M")
    generate_template_file(system_prompt)
    if not dry_run:
        upload_file(
            config.models["llama"].region,
            "template.json",
            bucket,
            "template.json",
            job_id + "/input",
        )
    generate_train_file(sample_data_file)
    if not dry_run:
        upload_file(
            config.models["llama"].region,
            "train.jsonl",
            bucket,
            "train.jsonl",
            job_id + "/input",
        )
    if not dry_run:
        run_estimator(
            config.models["llama"].model,
            config.models["llama"].version,
            sagemaker_execution_role,
            "s3://" + bucket + "/" + job_id + "/input",
            "s3://" + bucket + "/" + job_id + "/output",
            instance_type,
            config.models["llama"].region,
        )
