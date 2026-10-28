---
name: add-trainingdata-correction-config
description: Explore a pipeline's batch outputs, then write the correction prompt and declarative rules that BatchOutputCorrector applies to its training data
argument-hint: "<pipeline>"
---

Throughout, follow the CLAUDE.md files for this repository, and read `~/.claude/rules/INDEX.md` and the rules files it lists as matching, if they exist.
This skill writes correction config only.
The user wires that config into the pipeline's notebook and runs the correction.

## 1. Prepare

- The pipeline is `pipelines/$ARGUMENTS`.
  If no pipeline is given, ask which one.
- Find the pipeline's schema package from its `pyproject.toml` (e.g. `londonaicentre-opnoteschema` is `opnoteschema`), and read that package's `schema.py` and `prompt_datagen.txt` in `schemas/<package>/src/<package>/`.
- Read the `CorrectionRule` docstring in `lib/datagen/src/datagen/correction_rules.py` for the rule format.
- Other pipelines' `prompts/correction.txt` and `rules/correction.yaml` show the format in use, but their content belongs to their own schema, so do not carry it over.
- If the pipeline already has `prompts/correction.txt` or `rules/correction.yaml`, extend them rather than replacing them.
  Otherwise create the `prompts/` and `rules/` folders in the pipeline.
- Run every script below from the pipeline folder, as `uv run python ../../.claude/skills/add-trainingdata-correction-config/scripts/<script>.py --schema <package> ...`.
  The scripts read `anthropic_batch_job.jsonl.out` and write to `_scratch/corrections/`, which git ignores.
- Never Read the batch output file, as it can be gigabytes.
  Use the scripts, or short scripts of your own that print only counts and truncated examples.

## 2. Explore

- Run `corpus.py census` to count records by extraction status: `valid`, `invalid` (JSON failing the schema), `empty`, `max_tokens`, `has_output` (an `<output>` tag without JSON) and `prose`.
- Check corpus-wide patterns in the `valid` records with short scripts built on `Corpus.records`, `Corpus.note` and `Corpus.response` (with the `scripts` folder on `sys.path`), printing counts and at most three truncated examples each.
  Look for frequent enum values, fields that are set where a related field is null, descriptive fields not found verbatim in the note, and values with no support in the note (e.g. a date or category the note never states).
- Fan out reviewers across samples of the corpus:
  - Run `corpus.py sample valid`, and `corpus.py sample <status>` for each failure status, to write manifests of about 25 record IDs each.
  - Launch one reviewer per manifest, all in one message (or as one Workflow, with a structured schema for the returns), and no more than six in total.
  - Use the `sonnet` model for reviewers unless the notes need finer clinical judgement.
  - Brief each reviewer with the Brief section of `agent-brief.md` alone, filling in its placeholders from the rest of that file, and listing every issue already found so that it is not reported again.
  - Use only the reviewers' final reports, and never read their transcripts.
- Use `corpus.py show <recordId>...` to check any finding yourself.

## 3. Verify clinical conventions

- For each finding that rests on a clinical or coding convention (e.g. what counts as a separate procedure, or when a category may be recorded), supplement your own medical knowledge with authoritative sources found by WebSearch and WebFetch, such as NHS England clinical coding standards, OPCS-4, ICD-10, NCEPOD, NICE or Royal College guidance.
- Search once per topic, here rather than in each reviewer.
- Note the source next to the finding.

## 4. Triage and agree

- Assign each finding to one of:
  - the prompt, for records with no valid extraction, which Claude must create or fix;
  - a rule, for a mechanical fix to a valid extraction, decidable from its fields and a regular expression over the note;
  - none, for judgement calls on valid extractions (e.g. procedure choice or summaries), which neither the prompt nor the rules can safely make.
- The schema and datagen prompt are fixed, as the user may not be able to change them, so never triage a finding to a change in either.
  Where their wording is ambiguous, read it as written, and triage the finding to a rule only if every reading agrees, otherwise to none.
- Present the census, the findings with their counts and sources, and the triage to the user, and agree them before writing anything.

## 5. Write the prompt

- `correct_with_claude` sends only records with no valid extraction, so write the prompt for those alone.
- Use these sections: a title, a role, the inputs in `<note>` and `<original_response>` tags, numbered rules for each failure found, the original datagen prompt under REFERENCE INSTRUCTIONS as `{EXTRACTION_PROMPT}`, and OUTPUT last, with no code fences.
- Do not repeat what the datagen prompt already says.
- Test the prompt with `corpus.py test-prompt prompts/correction.txt <recordId>...` on up to ten records spread across the failure statuses.
  It corrects copies of the records with Claude Haiku, leaving the batch output untouched, and reports each record's status before and after.
  Each record takes about 25 seconds and uses the user's Claude quota.
- Revise the prompt until every tested record is valid and its correction is right.

## 6. Write the rules

- Write one rule per finding triaged to a rule, each with a `description` stating the convention it enforces.
- Pair `pattern` and `contains` with `type` wherever the value may be null or of another type, as they otherwise pass.
- Use `{value}` in a pattern to test whether the target value appears in the note.
- Order the rules so that any rule tidying up after others (e.g. clearing an emptied list) comes last.
- Run `dry_run.py rules/correction.yaml` to see how many records each rule changes, and whether a second pass would change any further.
- Run `dry_run.py rules/correction.yaml --rule <name> --show 10` for each rule, and check every change against its note for false positives, using `corpus.py show` for the full note.
  With many rules, fan out one reviewer per group of rules, as in section 2.
- Narrow any rule with a false positive, preferring to keep a value over wrongly changing it.

## 7. Report

- Report the files written and the dry run counts.
