---
name: add-docsynth-supplement-config
description: Find which of a pipeline's docsynth requirements its documents under-represent, then write the supplement that docsynth's Generator.supplement adds documents to the batch with
argument-hint: "<pipeline>"
---

Throughout, follow the CLAUDE.md files for this repository, and read `~/.claude/rules/INDEX.md` and the rules files it lists as matching, if they exist.
This skill writes the supplement only.
The user runs `Generator.supplement` in the pipeline's notebook, which generates the supplement's documents with Claude Code and adds them to the batch.

## 1. Prepare

- The pipeline is `pipelines/$ARGUMENTS`.
  If no pipeline is given, ask which one.
- Read the pipeline's `pipeline.yml` for the domain (`profile_selection.domain`), the prompt configuration and the output subdirectory.
- Find the domain's assets with `uv run python -c "from importlib.resources import files; print(files('docsynth') / 'assets' / '<domain>')"` from the pipeline folder, and read its `style.yml`, `content.yml`, `prompts/userprompt_docsynth.md`, structures and a few profiles.
- Read `types/supplements.py` and `Generator.supplement` in `generate.py`, in the same `docsynth` package, for the supplement format and how its prompts are built.
- If the pipeline already has `prompts/supplement.yaml`, extend it rather than replacing it.
  Otherwise create the `prompts/` folder in the pipeline.
- Run the script below from the pipeline folder, as `uv run python ../../.claude/skills/add-docsynth-supplement-config/scripts/documents.py ...`.
  It reads the documents in `output/<subdirectory>/` and writes to `_scratch/corrections/`, which git ignores.
- Never Read the documents folder whole, as it can hold thousands of documents.
  Use the script, or short scripts of your own that print only counts and truncated examples.

## 2. Explore

- Run `documents.py census` for each style and content option's sampled share against its configured probability.
  A sampled share more than about five points from its probability is itself a shortfall.
- Choose the requirements whose effect can be judged from the document text (e.g. the number of findings, the length, whether a prior study is compared), and leave out those too loose to judge reliably (e.g. an approximate share of redacted details).
- Fan out reviewers, one per requirement:
  - Run `documents.py sample --size 25 --batch 25 --requirement <section>=<option>` for each, which writes `documents-<section>-<option>_1.json` from the documents whose prompts asked for that option.
  - Launch the reviewers all in one message (or as one Workflow, with a structured schema for the returns), and no more than six in total.
    If more requirements need checking, give a reviewer several, each with its own manifest.
  - Use the `sonnet` model for reviewers unless the documents need finer clinical judgement.
  - Brief each reviewer with the Brief section of `agent-brief.md` alone, filling in its placeholders.
  - Use only the reviewers' final reports, and never read their transcripts.
- Use `documents.py show <doc_id>...` to check any finding yourself.

## 3. Verify clinical conventions

- For each finding that rests on a clinical or reporting convention (e.g. whether a phrase is a normal negative, or what a modality can show), supplement your own medical knowledge with authoritative sources found by WebSearch and WebFetch, such as NHS England, NICE or Royal College guidance.
- Search once per topic, here rather than in each reviewer.
- Note the source next to the finding.

## 4. Triage and agree

- For each requirement checked, its actual share is its sampled share times the share of reviewed documents that meet it.
  Documents that miss it count towards whatever option they show instead, as the reviewers report.
- Assign each requirement whose actual share falls short of its configured probability to one of:
  - a supplement entry, which pins that option, and any others it depends on (e.g. a comparison option for an overall change summary);
  - none, for criteria too loose to judge reliably or deviations a real document could plausibly have.
- Size each entry's count:
  - Its full count is the number of extra documents, all meeting the requirement, that would bring the share of documents meeting it back to its configured probability.
    With a probability `p` and `a` of the corpus's `N` documents meeting it, that is `(p × N − a) / (1 − p)`, rounded up.
  - Each document is part of a Claude Code call on the user's quota (five documents per call by default), so the total is limited to a budget, by default 5% of the corpus (e.g. 200 documents for 4000).
    Split the budget across the entries in proportion to their full counts, giving no entry more than its full count.
  - Adding documents for one option lowers the shares of the section's other options, so check the counts against each other.
- The docsynth assets are fixed, as the user may not be able to change them, so never triage a shortfall to a change in them.
  Where a requirement's wording is ambiguous, read it as written, and give it an entry only if every reading agrees that documents miss it.
- Present the findings with their counts and sources, and the entries with their full and budgeted counts and the budget, to the user, and agree them before writing anything.

## 5. Write the supplement

- Write `prompts/supplement.yaml`, with one entry per agreed shortfall:

  ```yaml
  supplements:
    - name: single-finding
      description: 40% of reports asked for one finding report one, a 10% share against 25% configured
      count: 100
      requirements:
        present_findings: one_finding
  ```

  - `name` is unique and kebab-case.
  - `description` states the shortfall, with its numbers.
  - `requirements` maps each pinned section of `style.yml` or `content.yml` to one of its option keys.
- Test it with `documents.py test-supplement prompts/supplement.yaml`, which generates one document per entry in `_scratch/corrections/supplement_test/`, leaving the batch untouched, and prints each document with its instructions.
- Check that each document meets its entry's pinned requirements.
  If one does not, pin further options that support it (e.g. a brief report length alongside one finding), and test again.

## 6. Report

- Report the file written, each entry's count, the total and the test results.
