from argparse import ArgumentParser
from dataclasses import dataclass

from oncollama.settings import Settings
from oncollama_assets.wrapper import OncoLlamaAssets
from docsynth.generate import Generator


@dataclass
class DocsynthArgs:
    batch: bool


def parse_CLI_args() -> DocsynthArgs:
    """Parse command line arguments"""
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Whether to process sample generation request as a batch job. Requires AWS credentials.",
    )
    return DocsynthArgs(**vars(parser.parse_args()))


def main() -> None:
    args: DocsynthArgs = parse_CLI_args()
    settings: Settings = Settings()
    if args.batch:
        Generator().generate(
            OncoLlamaAssets(), settings.US_BUCKET, settings.BEDROCK_EXECUTION_ROLE
        )
    else:
        Generator().generate(OncoLlamaAssets())
