from pathlib import Path

def generate_system_prompt(file: str = "systemprompt_datagen.md") -> str:
    base_dir = Path(__file__).parent
    ## CREATE SYSTEM PROMPT
    # schema
    with open(base_dir.parent / "schema/genomicextractmodel.py", "r") as f:
        schema_content = f.read()
        # TODO: see if converting raw text into actual Pydantic model object works better

    # examples
    examples_path = base_dir / "examples" 
    e1 = ""
    e2 = ""
    e3 = ""
    e4 = ""
    try:
        with open(examples_path / "e1.json", "r") as f:
            e1 = f.read()
        with open(examples_path / "e2.json", "r") as f:
            e2 = f.read()
        with open(examples_path / "e3.json", "r") as f:
            e3 = f.read()
        with open(examples_path / "e4.json", "r") as f:
            e4 = f.read()
    except FileNotFoundError as e:
        print(f"Warning: Could not load example file: {e}")

    # prompt
    with open(base_dir / file, "r") as f:
        system_prompt_template = f.read()

    # create full system prompt using replace instead of format to avoid issues with curly braces
    system_prompt = (
        system_prompt_template.replace("{schema_content}", schema_content)
        .replace("{e1}", e1)
        .replace("{e2}", e2)
        .replace("{e3}", e3)
        .replace("{e4}", e4)
    )
    return system_prompt