import sys
import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from sagemaker.jumpstart.estimator import JumpStartEstimator
from llm_assets.prompts import generate_system_prompt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import load_config
from utils.aws import upload_file


def parse_CLI_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        args : Namespace
            Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="anthropic_batch_job.jsonl.out",
        help="Name of the AWS Bedrock Anthropic batch inference output file containing sample data",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        action="store_true",
        help="Whether to simulate calling AWS endpoints",
    )
    arguments = parser.parse_args()
    return arguments

def generate_template_file() -> None:
    with open("template.json", "w") as outfile:
        print(json.dumps({
            "prompt": generate_system_prompt("systemprompt_finetune.md") + "\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n", "completion": "{response}"
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

def run_finetune_pipeline(
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

if __name__ == "__main__":
    args = parse_CLI_args()
    config = load_config()
    load_dotenv()

    job_id = "finetune/" + datetime.now().strftime("%Y-%m-%d-%H%M")

    generate_template_file()
    if(not args.dry_run):
        upload_file(
            config["llama"]["region"], 
            "template.json",
            os.getenv("BUCKET"),
            "template.json",
            job_id + "/input"
        )

    generate_train_file(args.file)
    if(not args.dry_run):
        upload_file(
            config["llama"]["region"],
            "train.jsonl",
            os.getenv("BUCKET"),
            "train.jsonl",
            job_id + "/input"
        )

    if(not args.dry_run):
        run_finetune_pipeline(
            config["llama"]["model"],
            config["llama"]["version"],
            os.environ["ROLE"],
            "s3://" + os.getenv("BUCKET") + "/" + job_id + "/input",
            "s3://" + os.getenv("BUCKET") + "/" + job_id + "/output",
            os.environ["INSTANCE_TYPE"],
            config["llama"]["region"]
        )  