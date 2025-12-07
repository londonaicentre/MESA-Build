from datagen.claude import BootstrapFileGenerator


def main() -> None:
    BootstrapFileGenerator.generate_from_existing_synthetic_data("output/oncollama")
