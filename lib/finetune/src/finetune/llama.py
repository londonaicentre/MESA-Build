import os
import json
from datetime import datetime

from finetune.config import Config
from sagemaker.jumpstart.estimator import JumpStartEstimator

from aws import upload_file

def generate_template_file(system_prompt) -> None:
    with open("template.json", "w") as outfile:
        print(json.dumps({
            "prompt": system_prompt + "\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n", "completion": "{response}"
        }), 
        file=outfile)

def generate_train_file(samples_input_file) -> None:
    with open("train.jsonl", "w") as outfile:
        with open(samples_input_file, "r") as infile:
            for line in infile:
                line = json.loads(line)
                text = line["modelOutput"]["content"][0]["text"].replace("```json", "").replace("```", "")
                try:
                    text = json.loads(text)
                    print(json.dumps({
                        "instruction": "Extract the given information into a structured schema.",
                        "context": text["content"],
                        "response": json.dumps(text["output"], ensure_ascii=True)
                    }), 
                    file=outfile)
                except Exception:
                    pass

def run_estimator(
        model_id, 
        model_version, 
        role, 
        input_path, 
        output_path, 
        instance_type, 
        region, 
        instruction_tuned=True, 
        quantized=True
    ) -> None:
    estimator = JumpStartEstimator(
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
    )
    estimator.fit({'training': input_path})

def run_finetune(
    system_prompt,
    sample_data_file, 
    bucket, 
    sagemaker_execution_role, 
    instance_type, 
    dry_run = False
):
    config: Config = Config()
    
    job_id = "finetune/" + datetime.now().strftime("%Y-%m-%d-%H%M")

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