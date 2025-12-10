# SchemaLlama Assets

Static per-pipeline assets.

## Structure

Each pipeline assets folder is expected to have the following structure:

```text
📁 ASSETS
├── <pipeline>/
├──── examples/            # A set of unstructured input to structured output examples (datagen)
├──── profiles/            # Profiles that represent condition topography/morphology and biomarkers (docsynth)
├──── prompts/             # Prompt templates
├──── structure/           # Synthetic documents that mimic real clinical document 'structures' (docsynth)
├──── content.yml          # Probabilistic sampling file for content requirements (docsynth)
├──── style.yml            # Probabilistic sampling file for style requirements (docsynth)
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

- `profiles/`: Should contain a set of profile files formatted according to the structure defined in [`Profiles`](../lib/types/src/schemallama_types/assets/profile.py#L19). This model also provides insight on extra elements and their structure.

- `prompts/`: Should contain a set of prompt files that follow the naming convention: `<system | user>prompt_<primary lib name>.md`.

- `structure`: Should contain a set of structure files as text documents

- `content.yml`: The content sampling file should be formatted according to the structure defined in [`Content`](../lib/types/src/schemallama_types/assets/sampling.py#L134). This model also provides insight on extra elements and their structure.

- `style.yml`: The style sampling file should be formatted according to the structure defined in [`Style`](../lib/types/src/schemallama_types/assets/sampling.py#L69). This model also provides insight on extra elements and their structure.

- `wrapper.py`: Should contain a class that extends [`SchemaLlamaAssets`](../lib/types/src/schemallama_types/assets/wrapper.py#L15), providing implementations for abstract methods. This class can then be passed to relevant [`lib`](../lib/) packages from corresponding [`pipelines`](../pipelines/).
