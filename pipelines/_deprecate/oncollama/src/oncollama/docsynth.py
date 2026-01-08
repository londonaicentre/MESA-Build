from argparse import ArgumentParser
from dataclasses import dataclass

from oncollama.settings import Settings
from docsynth.generate import Generator
from docsynth.assets.oncology.wrapper import OncologyAssets


@dataclass
class DocsynthArgs:
    batch: bool
    extract: bool


def parse_CLI_args() -> DocsynthArgs:
    """Parse command line arguments"""
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Whether to process sample generation request as a batch job. Requires AWS credentials.",
    )
    parser.add_argument(
        "-e",
        "--extract",
        action="store_true",
        help="Generate documents from the outputs of a previous Bedrock batch inference run, rather than generating them live.",
    )
    return DocsynthArgs(**vars(parser.parse_args()))


def main() -> None:
    args: DocsynthArgs = parse_CLI_args()
    settings: Settings = Settings()
    if args.batch:
        Generator().generate(
            OncologyAssets(), settings.BUCKET, settings.BEDROCK_EXECUTION_ROLE
        )
    elif args.extract:
        Generator().extract_batch_output()
    else:
        Generator().generate(OncologyAssets())
