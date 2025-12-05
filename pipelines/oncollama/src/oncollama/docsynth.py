from oncollama_assets.wrapper import OncoLlamaAssets
from docsynth.generate import Generator


def main() -> None:
    Generator().generate(OncoLlamaAssets())
