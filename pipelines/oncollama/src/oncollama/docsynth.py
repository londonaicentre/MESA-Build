from oncollama_assets import OncoLlamaAssets
from docsynth.generate import Generator


def main() -> None:
    Generator().generate(OncoLlamaAssets())
