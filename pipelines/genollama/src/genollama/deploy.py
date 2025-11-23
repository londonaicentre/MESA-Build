from argparse import ArgumentParser
from dataclasses import dataclass

from deploy.llama import Deployer
from genollama.settings import Settings


@dataclass
class DeployArgs:
    path: str
    command: str


def parse_CLI_args() -> DeployArgs:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path within S3 bucket to the zipped weights of the model to deploy",
    )
    parser.add_argument("command", choices=["up", "down"])
    return DeployArgs(**vars(parser.parse_args()))


def main() -> None:
    args: DeployArgs = parse_CLI_args()
    settings: Settings = Settings()
    deployer: Deployer = Deployer(settings.IMAGE, settings.INSTANCE_TYPE)

    if args.command == "up":
        deployer.run_deploy_up(
            settings.BUCKET,
            args.path,
            settings.SAGEMAKER_EXECUTION_ROLE,
        )
    elif args.command == "down":
        deployer.run_deploy_down()
