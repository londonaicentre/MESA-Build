from argparse import ArgumentParser
from dataclasses import dataclass

from oncollama.settings import Settings
from oncollama_assets.wrapper import OncoLlamaAssets
from finetune.llama import FineTuner


@dataclass
class FinetuneArgs:
    folder: str
    dry_run: bool


def parse_CLI_args() -> FinetuneArgs:
    """Parse command line arguments"""
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="samples_sonnet4",
        help="Name of the folder containing sample data processed from AWS Bedrock Anthropic batch inference output",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        action="store_true",
        help="Whether to simulate calling AWS endpoints",
    )
    return FinetuneArgs(**vars(parser.parse_args()))


def main() -> None:
    args: FinetuneArgs = parse_CLI_args()
    settings: Settings = Settings()
    fine_tuner: FineTuner = FineTuner(settings.TRAINING_INSTANCE_TYPE)
    oncollama_assets: OncoLlamaAssets = OncoLlamaAssets()
    fine_tuner.run_finetune(
        oncollama_assets.load_system_prompt("systemprompt_finetune.md"),
        args.folder,
        settings.BUCKET,
        settings.SAGEMAKER_EXECUTION_ROLE,
        args.dry_run,
    )
