from oncollama.settings import Settings
from oncollama_assets.wrapper import OncoLlamaAssets
from docsynth.generate import Generator


def main() -> None:
    settings: Settings = Settings()
    Generator().generate(
        OncoLlamaAssets(), settings.US_BUCKET, settings.BEDROCK_EXECUTION_ROLE
    )
