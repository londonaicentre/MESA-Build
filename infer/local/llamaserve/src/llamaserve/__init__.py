from argparse import ArgumentParser

from llamaserve.llamaserve_types import LlamaServeArguments
from llamaserve.llamaserve import LlamaServe


def parse_args() -> LlamaServeArguments:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print debug output",
    )
    return LlamaServeArguments(**vars(parser.parse_args()))


def run() -> None:
    args: LlamaServeArguments = parse_args()
    llamaServe: LlamaServe = LlamaServe(args.verbose)
    if llamaServe.unpack():
        llamaServe.serve()
