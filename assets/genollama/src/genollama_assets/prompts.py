from importlib.resources import files
from typing import Any


def load(folder: str, file: str) -> str:
    """Load a file from within the package

    Args:
        folder (str): Folder containing the file
        file (str): Name of the file

    Returns:
        str: File contents

    """
    return files("genollama_assets").joinpath(f"{folder}/{file}").read_text()


def generate_system_prompt(file: str = "systemprompt_datagen.md") -> str:
    """Create a system prompt

    Args:
        file (str, optional): The template file to use for the system prompt.
            Defaults to the datagen system prompt.

    Returns:
        str: The system prompt

    """

    # schema
    schema_content: str = load(".", "genollama_assets_types.py")

    # examples
    examples_path: str = "examples"
    e1: str = ""
    e2: str = ""
    e3: str = ""
    e4: str = ""
    try:
        e1 = load(examples_path, "e1.json")
        e2 = load(examples_path, "e2.json")
        e3 = load(examples_path, "e3.json")
        e4 = load(examples_path, "e4.json")
    except FileNotFoundError as e:
        print(f"Warning: Could not load example file: {e}")

    # prompt
    system_prompt_template: str = load("templates", file)

    # create full system prompt using replace instead of format to avoid issues with curly braces
    system_prompt: str = (
        system_prompt_template.replace("{schema_content}", schema_content)
        .replace("{e1}", e1)
        .replace("{e2}", e2)
        .replace("{e3}", e3)
        .replace("{e4}", e4)
    )
    return system_prompt


def generate_bootstrap_user_prompt(instructions: str) -> str:
    """Create a user prompt for bootstrap file generation

    Args:
        instructions (str): Instruction to tailor the bootstrap file output

    Returns:
        str: The user prompt

    """
    user_prompt: str = f"""Please now generate 20 rows according to the above instructions as a CSV file. These rows should {instructions}. While conforming to these instructions, please also ensure that rows are varied, and represent a range of different report types and styles."""
    return user_prompt


def generate_datagen_user_prompt(row: dict[str, Any]) -> str:
    """Create a user prompt for sample file generation

    Args:
        row (dict): The bootstrap file row from which to generate data to tailor the sample

    Returns:
        str: The user prompt

    """
    user_prompt: str = f"""Please generate a genomic laboratory report based on the following test scenario:

        Test Type: {row["test_type"]}
        Test Details: {row["test_details"]}
        Result Entities: {row["result_entities"]}
        Result Description: {row["result_description"]}
        Clinical Context: {row["clinical_context"]}
        Disease Context: {row["disease_context"]}
        Family History: {row["family_history"]}
        Test Subject: {row["test_subject"]}
        Clinical Implications: {row["clinical_implications"]}
        Recommendations: {row["recommendations"]}
        Report Style: {row["report_style"]}

        Generate a realistic genomic laboratory report incorporating all these details.
        Then extract the information into the structured schema format."""
    return user_prompt
