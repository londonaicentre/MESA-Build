import argparse
import os

from dotenv import load_dotenv

from deploy.llama import run_deploy_up, run_deploy_down


def parse_CLI_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path within S3 bucket to the zipped weights of the model to deploy",
    )
    parser.add_argument(
        "command", 
        choices=["up", "down"]
    )
    arguments = parser.parse_args()
    return arguments

def main():
    args = parse_CLI_args()
    load_dotenv()
    
    if(args.command == "up"):
        run_deploy_up(
            os.getenv("BUCKET"), 
            args.path, 
            os.getenv("SAGEMAKER_EXECUTION_ROLE"),
            os.getenv("IMAGE"), 
            os.getenv("INSTANCE_TYPE")
        )
    elif(args.command == "down"):
        run_deploy_down(os.getenv("IMAGE"))