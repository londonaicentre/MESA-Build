import argparse
import os

from dotenv import load_dotenv

from genollama_assets.prompts import generate_system_prompt, generate_datagen_user_prompt
from genollama_assets.genollama_assets_types import GenomicTestReport
from claudedatagen import run_datagen, run_backfill, run_batch


def parse_CLI_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        args : Namespace
            Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_name",
        type=str,
        default="sonnet4",
        help="Name of model to use, eg sonnet4 or opus4",
    )
    parser.add_argument(
        "-s",
        "--sample_size",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default="bootstrap.csv",
        help="Path to file with sample template",
    )
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Whether to process sample generation request as a batch job. Requires AWS credentials.",
    )
    parser.add_argument(
        "-f",
        "--backfill",
        type=bool,
        default=False,
        help="Generate samples for skipped indices",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    # Read the arguments from CLI
    args = parse_CLI_args()
    # load api key
    load_dotenv()
    BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY")
    system_prompt = generate_system_prompt()
    if args.batch:
        run_batch(
            system_prompt,
            generate_datagen_user_prompt,
            args.model_name,
            args.template,
            args.sample_size,
            os.getenv("BUCKET"),
            os.getenv("BEDROCK_EXECUTION_ROLE"),
        )
    elif args.backfill:
        run_backfill(
            system_prompt,
            generate_datagen_user_prompt,
            args.model_name,
            args.template,
            args.sample_size,
            os.getenv("BEDROCK_API_KEY"),
            GenomicTestReport,
        )
    else:
        run_datagen(
            system_prompt,
            generate_datagen_user_prompt,
            args.model_name,
            args.template,
            args.sample_size,
            os.getenv("BEDROCK_API_KEY"),
            GenomicTestReport,
        )
