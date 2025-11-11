from importlib.resources import files

def load(folder: str, file: str) -> str:
    return files("llm_assets").joinpath(f"{folder}/{file}").read_text()

def generate_system_prompt(file: str = "systemprompt_datagen.md") -> str:
    ## CREATE SYSTEM PROMPT
    # schema
    schema_content = load(".", "llm_assets_types.py")
    # TODO: see if converting raw text into actual Pydantic model object works better

    # examples
    examples_path = "examples" 
    e1 = ""
    e2 = ""
    e3 = ""
    e4 = ""
    try:
        e1 = load(examples_path, "e1.json")
        e2 = load(examples_path, "e2.json")
        e3 = load(examples_path, "e3.json")
        e4 = load(examples_path, "e4.json")
    except FileNotFoundError as e:
        print(f"Warning: Could not load example file: {e}")

    # prompt
    system_prompt_template = load("templates", file)

    # create full system prompt using replace instead of format to avoid issues with curly braces
    system_prompt = (
        system_prompt_template.replace("{schema_content}", schema_content)
        .replace("{e1}", e1)
        .replace("{e2}", e2)
        .replace("{e3}", e3)
        .replace("{e4}", e4)
    )
    return system_prompt