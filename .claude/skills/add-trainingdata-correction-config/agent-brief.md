## Brief

You are reviewing LLM-generated structured extractions, used as training data, against the notes they were extracted from.

- The extraction schema is `{SCHEMA_PATH}`, and the extraction instructions are `{DATAGEN_PROMPT_PATH}`.
- Your records are listed in `{MANIFEST}`.
- From `{PIPELINE_DIR}`, view them with `uv run python ../../.claude/skills/add-trainingdata-correction-config/scripts/corpus.py --schema {SCHEMA_PACKAGE} show <recordId>...`, several at a time.
  Add `--chars <n>` to see more of a long note.
- Never open the batch output file directly.

{FOCUS}

These issues are already known, so do not report them again:

{KNOWN_ISSUES}

Report only issues you have verified against the note text.
For each, give the number of your records it affects (n/N), one or two record IDs, and a quote from the note of no more than 20 words.
Group issues by kind, most frequent first, in no more than 400 words.

## Focus

Replace `{FOCUS}` with the text matching the reviewer's manifest.

- `valid`: Find systematic errors in the extractions: values the note does not support, fields that break the schema's own rules, and wrong or missed codes or categories.
- A failure status: Find why each response has no valid extraction (e.g. it declines, comments on the note, continues the note or breaks the schema), whether the note is the kind of document the schema is for, and what a correct response would be.
