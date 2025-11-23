from argparse import ArgumentParser
from dataclasses import dataclass

from genollama.settings import Settings
from genollama_assets.prompts import (
    generate_system_prompt,
    generate_datagen_user_prompt,
)
from genollama_assets.genollama_assets_types import GenomicTestReport
from datagen.claude import run_sample_generation, run_backfill, run_batch_inference


@dataclass
class DatagenArgs:
    model_name: str
    sample_size: int
    template: str
    batch: bool
    backfill: bool


def parse_CLI_args() -> DatagenArgs:
    """Parse command line arguments"""
    parser: ArgumentParser = ArgumentParser()
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
    return DatagenArgs(**vars(parser.parse_args()))


def main() -> None:
    # Read the arguments from CLI
    args: DatagenArgs = parse_CLI_args()
    # load api key
    settings: Settings = Settings()
    system_prompt = generate_system_prompt()
    if args.batch:
        run_batch_inference(
            system_prompt,
            generate_datagen_user_prompt,
            args.model_name,
            args.template,
            args.sample_size,
            settings.US_BUCKET,
            settings.BEDROCK_EXECUTION_ROLE,
        )
    elif args.backfill:
        run_backfill(
            system_prompt,
            generate_datagen_user_prompt,
            args.model_name,
            args.template,
            args.sample_size,
            settings.BEDROCK_API_KEY,
            GenomicTestReport,
        )
    else:
        run_sample_generation(
            system_prompt,
            generate_datagen_user_prompt,
            args.model_name,
            args.template,
            args.sample_size,
            settings.BEDROCK_API_KEY,
            GenomicTestReport,
        )
