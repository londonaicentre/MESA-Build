import json
from datetime import datetime
from typing import Any

from sagemaker.jumpstart.estimator import JumpStartEstimator

from finetune.config import Config
from utils.aws import upload_file

def generate_template_file(system_prompt: str) -> None:
    with open("template.json", "w") as outfile:
        print(json.dumps({
            "prompt": system_prompt + "\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n", "completion": "{response}"
        }), 
        file=outfile)

def generate_train_file(samples_input_file: str) -> None:
    with open("train.jsonl", "w") as outfile:
        with open(samples_input_file, "r") as infile:
            for line in infile:
                parsed_line: dict[str, Any] = json.loads(line)
                text: str = parsed_line["modelOutput"]["content"][0]["text"].replace("```json", "").replace("```", "")
                try:
                    parsed_text: dict[str, Any] = json.loads(text)
                    print(json.dumps({
                        "instruction": "Extract the given information into a structured schema.",
                        "context": parsed_text["content"],
                        "response": json.dumps(parsed_text["output"], ensure_ascii=True)
                    }), 
                    file=outfile)
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
        quantized: bool = True
    ) -> None:
    estimator: JumpStartEstimator = JumpStartEstimator(
        model_id=model_id,
        model_version=model_version,
        role=role,
        disable_output_compression=True,
        output_path=output_path,
        instance_type=instance_type,
        region=region,
        environment={'accept_eula': 'true'},
    )
    estimator.set_hyperparameters(
        instruction_tuned='True' if instruction_tuned else 'False',
        chat_dataset='False' if instruction_tuned else 'True',
        int8_quantization='True' if quantized else 'False',
        enable_fsdp='False' if quantized else 'True'
    )  # type: ignore[no-untyped-call]
    estimator.fit({'training': input_path})

def run_finetune(
    system_prompt: str,
    sample_data_file: str,  
    bucket: str, 
    sagemaker_execution_role: str, 
    instance_type: str, 
    dry_run: bool = False
) -> None:
    config: Config = Config()
    
    job_id: str = "finetune/" + datetime.now().strftime("%Y-%m-%d-%H%M")

    generate_template_file(system_prompt)
    if(not dry_run):
        upload_file(
            config.models["llama"].region, 
            "template.json",
            bucket,
            "template.json",
            job_id + "/input"
        )

    generate_train_file(sample_data_file)
    if(not dry_run):
        upload_file(
            config.models["llama"].region,
            "train.jsonl",
            bucket,
            "train.jsonl",
            job_id + "/input"
        )

    if(not dry_run):
        run_estimator(
            config.models["llama"].model,
            config.models["llama"].version,
            sagemaker_execution_role,
            "s3://" + bucket + "/" + job_id + "/input",
            "s3://" + bucket + "/" + job_id + "/output",
            instance_type,
            config.models["llama"].region
        )  