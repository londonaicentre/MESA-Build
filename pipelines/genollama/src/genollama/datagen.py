from argparse import ArgumentParser
from dataclasses import dataclass

from genollama.settings import Settings
from genollama_assets.wrapper import GenoLlamaAssets
from genollama_assets.schema import GenomicTestReport
from datagen.claude import SampleGenerator


@dataclass
class DatagenArgs:
    model_name: str
    sample_size: int
    template: str
    batch: bool
    backfill: bool
    extract: bool


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
    parser.add_argument(
        "-e",
        "--extract",
        action="store_true",
        help="Extract generated samples from a Bedrock batch output file.",
    )
    return DatagenArgs(**vars(parser.parse_args()))


def main() -> None:
    args: DatagenArgs = parse_CLI_args()
    settings: Settings = Settings()
    genollama_assets: GenoLlamaAssets = GenoLlamaAssets()
    system_prompt: str = genollama_assets.load_system_prompt()
    sample_generator: SampleGenerator = SampleGenerator(
        system_prompt,
        genollama_assets.load_datagen_user_prompt,
        GenomicTestReport,
        args.model_name,
        args.template,
        settings.BEDROCK_API_KEY,
    )
    if args.batch:
        sample_generator.generate_via_batch(
            args.sample_size,
            settings.US_BUCKET,
            settings.BEDROCK_EXECUTION_ROLE,
        )
    elif args.extract:
        sample_generator.extract_batch_output()
    elif args.backfill:
        sample_generator.run_backfill(
            args.sample_size,
        )
    else:
        sample_generator.generate(
            args.sample_size,
        )
