from argparse import ArgumentParser
from dataclasses import dataclass

from genollama.settings import Settings
from genollama_assets.prompts import (
    generate_system_prompt,
    generate_bootstrap_user_prompt,
)
from datagen.claude import run_bootstrap_file_generation


@dataclass
class BootstrapArgs:
    model_name: str
    instruction: str


def parse_CLI_args() -> BootstrapArgs:
    """Parse command line arguments"""
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        default="sonnet4",
        help="Name of model to use, eg sonnet4 or opus4",
    )
    parser.add_argument(
        "-i",
        "--instruction",
        type=str,
        required=True,
        help="Tailor the bootstrap file output, e.g. point the batch at a type of test, a disease area, a particular proband pattern, a report style, or any other variable",
    )
    return BootstrapArgs(**vars(parser.parse_args()))


def main() -> None:
    # Read the arguments from CLI
    args: BootstrapArgs = parse_CLI_args()
    # load api key
    settings: Settings = Settings()
    run_bootstrap_file_generation(
        generate_system_prompt("systemprompt_bootstrap.md"),
        generate_bootstrap_user_prompt,
        args.instruction,
        args.model_name,
        settings.BEDROCK_API_KEY,
    )
