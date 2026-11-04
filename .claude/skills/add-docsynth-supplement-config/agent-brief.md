## Brief

You are checking whether LLM-generated synthetic clinical documents, used to create training data, meet one requirement of the prompts they were generated from.

- The documents' style and content requirements are sampled from `{STYLE_PATH}` and `{CONTENT_PATH}`, and their profiles from `{PROFILES_PATH}`.
- Every document listed in `{MANIFEST}` was asked to meet this requirement, from the `{SECTION}` section: {REQUIREMENT}
- From `{PIPELINE_DIR}`, view the documents with `uv run python ../../.claude/skills/add-docsynth-supplement-config/scripts/documents.py show <doc_id>...`, several at a time, which prints each document's requirements and profile, then its content.
  Add `--chars <n>` to see more of a long document.
- Never open the documents folder directly.

Judge each document against the requirement as written, and against its own profile where the requirement refers to it (e.g. the profile's possible findings).
Where a document misses the requirement, say which of the section's other options it matches instead, from:

{OTHER_OPTIONS}

Report, in no more than 300 words:

- the number of documents that meet the requirement (n/N);
- for those that miss it, the number matching each other option, with one or two doc IDs and a quote from the document of no more than 20 words;
- any document you could not judge, and why.
