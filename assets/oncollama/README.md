# OncoLlama Assets

Assets for OncoLlama: Generating high fidelity synthetic cancer letters, and fine-tuning LLMs for structured data extraction.

## Features

This package primarily exposes the **OncoLlama schema** for runtime output validation. Use it to:

- **Validate LLM outputs** against the expected OncoLlama structure
- **Generate JSON schemas** for API contracts and documentation
- **Parse and validate** extracted oncology data at runtime

### Schema Access

Access the Pydantic model and its JSON schema:

```python
from oncollama_assets.schema import OncoLlamaModel

# Get the JSON schema for validation
schema = OncoLlamaModel.model_json_schema()

# Parse and validate deserialised output
data = OncoLlamaModel.model_validate(llm_output)
# Parse and validate deserialised json string
data = OncoLlamaModel.model_validate_json(llm_output)
```

### System Prompts

Load system prompts with the schema automatically injected:

```python
from oncollama_assets.wrapper import OncoLlamaAssets

assets = OncoLlamaAssets()

# Load inference system prompt (default)
system_prompt = assets.load_system_prompt()

# Or specify a different prompt template
system_prompt = assets.load_system_prompt("systemprompt_finetune.md")
```

Available prompt templates:
- `systemprompt_infer.md` - For inference (default)
- `systemprompt_finetune.md` - For fine-tuning
- `systemprompt_datagen.md` - For data generation

### Wrapper Class (Internal Use)

The `OncoLlamaAssets` wrapper class also provides testing and internal release mechanisms.

## Structure

```text
📁 ONCOLLAMA_ASSETS
├── prompts/             # Prompt templates
├── schema.py            # Pydantic model for specifying expected OncoLlama output structure
├── wrapper.py           # Wrapper class for internal testing and release mechanisms
```

## License

This project uses a proprietary license (see [LICENSE](LICENSE.md)).
