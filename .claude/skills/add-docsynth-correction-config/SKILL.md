---
name: add-docsynth-correction-config
description: Explore a pipeline's docsynth documents, then write the review prompt that docsynth's Corrector judges each document against its generation prompt with
argument-hint: "<pipeline>"
---

Throughout, follow the CLAUDE.md files for this repository, and read `~/.claude/rules/INDEX.md` and the rules files it lists as matching, if they exist.
This skill writes the review prompt only.
The user wires it into the pipeline's notebook and runs the review and corrections, which write `rules/document_correction.yaml`.

## 1. Prepare

- The pipeline is `pipelines/$ARGUMENTS`.
  If no pipeline is given, ask which one.
- Read the pipeline's `pipeline.yml` for the domain (`profile_selection.domain`), the prompt configuration and the output subdirectory.
- Find the domain's assets with `uv run python -c "from importlib.resources import files; print(files('docsynth') / 'assets' / '<domain>')"` from the pipeline folder, and read its `style.yml`, `content.yml`, `prompts/userprompt_docsynth.md`, structures and a few profiles.
- Read `config/document_correction_format.txt` and `correct.py` in the same `docsynth` package, for the output format appended to the review prompt and for how the rules are applied.
- If the pipeline already has `prompts/document_correction.txt`, extend it rather than replacing it.
  Otherwise create the `prompts/` folder in the pipeline.
- Run the script below from the pipeline folder, as `uv run python ../../.claude/skills/add-docsynth-correction-config/scripts/documents.py ...`.
  It reads the documents in `output/<subdirectory>/` and writes to `_scratch/corrections/`, which git ignores.
- Never Read the documents folder whole, as it can hold thousands of documents.
  Use the script, or short scripts of your own that print only counts and truncated examples.

## 2. Explore

- Check corpus-wide patterns with short scripts over the document JSON files, printing counts and at most three truncated examples each.
  Look for wording from the generation prompt reused in documents (e.g. requirement names), lengths that do not match the length requirement, profiles whose expected outcome the document contradicts, and leftover preamble or tags.
- Fan out reviewers across samples of the corpus:
  - Run `documents.py sample` to write manifests of about 25 doc IDs each.
  - Launch one reviewer per manifest, all in one message (or as one Workflow, with a structured schema for the returns), and no more than six in total.
  - Use the `sonnet` model for reviewers unless the documents need finer clinical judgement.
  - Brief each reviewer with the Brief section of `agent-brief.md` alone, filling in its placeholders, and listing every issue already found so that it is not reported again.
  - Use only the reviewers' final reports, and never read their transcripts.
- Use `documents.py show <doc_id>...` to check any finding yourself.

## 3. Verify clinical conventions

- For each finding that rests on a clinical or reporting convention (e.g. whether a phrase is a normal negative, or what a modality can show), supplement your own medical knowledge with authoritative sources found by WebSearch and WebFetch, such as NHS England, NICE or Royal College guidance.
- Search once per topic, here rather than in each reviewer.
- Note the source next to the finding.

## 4. Triage and agree

- Assign each finding to one of:
  - `replace`, where removing or rewriting one exact span fixes the document (e.g. a section the generation prompt says to leave out);
  - `regenerate`, where the fix spans the document (e.g. the wrong number of findings, or the wrong length);
  - none, for criteria too loose to judge reliably (e.g. an approximate share of redacted details) or deviations a real document could plausibly have.
- The docsynth assets are fixed, as the user may not be able to change them, so never triage a finding to a change in them.
  Where a requirement's wording is ambiguous, read it as written, and triage the finding to `replace` or `regenerate` only if every reading agrees, otherwise to none.
- Present the findings with their counts and sources, and the triage, to the user, and agree them before writing anything.

## 5. Write the prompt

- The prompt is the system prompt for every review, with each document's generation prompt and content in the user message, so write it for the domain rather than for any one document.
- Use these sections: a title, a role, numbered checks for each finding triaged to `replace` or `regenerate`, each saying which action fixes it, and the criteria to ignore.
- Do not repeat the output format, which docsynth appends, or what the generation prompts already say.
- Test the prompt with `documents.py test-prompt prompts/document_correction.txt <doc_id>...` on up to ten documents, covering each finding and documents with none.
  It reviews and corrects copies of the documents, leaving the originals untouched, and reports each document's rules and the diff of its corrections.
  Each document takes a Claude Sonnet review, and a Claude Haiku regeneration if it gets a `regenerate` rule, using the user's Claude quota.
- Revise the prompt until every tested document gets the rules it should, and none it should not.

## 6. Report

- Report the file written and the test results.
