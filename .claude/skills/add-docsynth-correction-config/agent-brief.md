## Brief

You are reviewing LLM-generated synthetic clinical documents, used to create training data, against the prompts they were generated from.

- The documents' style and content requirements are sampled from `{STYLE_PATH}` and `{CONTENT_PATH}`, and their profiles from `{PROFILES_PATH}`.
- Your documents are listed in `{MANIFEST}`.
- From `{PIPELINE_DIR}`, view them with `uv run python ../../.claude/skills/add-docsynth-correction-config/scripts/documents.py show <doc_id>...`, several at a time, which prints each document's requirements and profile, then its content.
  Add `--chars <n>` to see more of a long document.
- Never open the documents folder directly.

Find where documents break a requirement or the profile they were generated from: missing or extra sections, the wrong number of findings, lengths or styles that do not match, details the profile contradicts, wording reused from the requirements, and leftover preamble.
Judge each document against its own requirements, as they differ from document to document.

These issues are already known, so do not report them again:

{KNOWN_ISSUES}

Report only issues you have verified against the document text.
For each, give the number of your documents it affects (n/N), one or two doc IDs, the requirement it breaks, and a quote from the document of no more than 20 words.
Say whether replacing one exact span would fix it, or the document needs regenerating.
Group issues by kind, most frequent first, in no more than 400 words.
