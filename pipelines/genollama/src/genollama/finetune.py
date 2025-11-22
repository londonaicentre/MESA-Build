import os
import argparse

from dotenv import load_dotenv

from genollama_assets.prompts import generate_system_prompt
from finetune.llama import run_finetune

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

def main():
    args = parse_CLI_args()
    load_dotenv()
    run_finetune(
        generate_system_prompt("systemprompt_finetune.md"),
        args.file, 
        os.getenv("BUCKET"), 
        os.environ["SAGEMAKER_EXECUTION_ROLE"],
        os.environ["INSTANCE_TYPE"], 
        args.dry_run
    )