# SchemaLlama Assets

Static per-pipeline assets.

## Structure

Each pipeline assets folder is expected to have the following structure:

```text
📁 ASSETS
├── <pipeline>/
├──── examples/            # A set of unstructured input to structured output examples (datagen)
├──── prompts/             # Prompt templates
├──── schema.py            # Pydantic model for specifying expected LLM output structure
├──── wrapper.py           # Wrapper class for serving, and operating on, stored assets
```

Additional structural details:

- `examples`: Should contain the examples as a set of json files with the structure:

    ```json
    {
        "content": "unstructured input",
        "output": {
            "structured output": "structured output"
        }
    }
    ```

- `prompts/`: Should contain a set of prompt files that follow the naming convention: `<system | user>prompt_<primary lib name>.md`.

- `wrapper.py`: Should contain a class that extends [`SchemaLlamaAssets`](../lib/types/src/schemallama_types/assets/wrapper.py#L15), providing implementations for abstract methods. This class can then be passed to relevant [`lib`](../lib/) packages from corresponding [`pipelines`](../pipelines/).
