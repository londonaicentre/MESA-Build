from argparse import ArgumentParser
from dataclasses import dataclass

from genollama.settings import Settings
from genollama_assets.wrapper import GenoLlamaAssets
from datagen.claude import BootstrapFileGenerator


@dataclass
class BootstrapArgs:
    model_name: str
    instruction: str
    backup: bool


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
    parser.add_argument(
        "-s",
        "--backup",
        action="store_true",
        help="Whether to store the generated bootstrap file in S3 (requires credentials)",
    )
    return BootstrapArgs(**vars(parser.parse_args()))


def main() -> None:
    args: BootstrapArgs = parse_CLI_args()
    settings: Settings = Settings()
    genollama_assets: GenoLlamaAssets = GenoLlamaAssets()
    BootstrapFileGenerator.generate(
        genollama_assets.load_system_prompt("systemprompt_bootstrap.md"),
        genollama_assets.load_bootstrap_user_prompt,
        args.instruction,
        args.model_name,
        settings.BEDROCK_API_KEY,
        settings.US_BUCKET if args.backup else "",
    )
