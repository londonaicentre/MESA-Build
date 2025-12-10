# OncoLlama Assets

Assets for OncoLlama: Generating high fidelity synthetic cancer letters, and fine-tuning LLMs for structured data extraction.

## Structure

```text
📁 ONCOLLAMA_ASSETS
├── profiles/            # Profiles that represent condition topography/morphology and biomarkers
├── prompts/             # Prompt templates
├── structure/           # Synthetic documents that mimic real oncology document 'structures'
├── content.yml          # Probabilistic sampling file for content requirements 
├── style.yml            # Probabilistic sampling file for style requirements
├── schema.py            # Pydantic model for specifying expected OncoLlama output structure
├── wrapper.py           # Wrapper class for serving, and operating on, stored OncoLlama assets
```

## License

This project uses a proprietary license (see [LICENSE](LICENSE.md)).
