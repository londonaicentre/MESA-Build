# mesa-build

## Python versions

Every package must pin its Python version in a `.python-version` file.
Without one, CI's uv picks the newest CPython, and `uv sync` fails when a dependency has no wheel for it yet and needs a compiler the runner lacks (e.g. tiktoken, a litellm dependency, needs Rust on CPython 3.15).

Review this note once tiktoken publishes CPython 3.15 wheels (first seen 2026-10-20), as the pin may then only guard against the next Python release.

## PyPI trusted publishers

Each package's publish workflow (e.g. `conditionschema-build-and-publish.yml`) calls the reusable `build-and-publish.yml`, and PyPI matches a trusted publisher on the token's `job_workflow_ref`, which names the reusable workflow.
So a new package's pending trusted publisher names the workflow `build-and-publish.yml` (environment `pypi`), never the caller.
A pending publisher that names the caller is skipped, and the upload falls back to the active publisher the existing packages share, failing with `Non-user identities cannot create new projects`.

## Schema docs

A schema's README and diagram come from `schemas-build-docs.yml`, which has no push or PR trigger and is run locally with act from the repository root:

```sh
mkdir -p schemas/<name>/_assets
GITHUB_TOKEN="$(gh auth token)" act -b workflow_dispatch -W .github/workflows/schemas-build-docs.yml -s GITHUB_TOKEN \
  --input assets_dir=schemas/<name> \
  --input "description=<README sentence>" \
  --input base_model_path=<name>.schema \
  --input base_model=<top-level model>
```

- `-s GITHUB_TOKEN` with no value reads the token from the environment, so it never appears on the command line.
- `-b` binds the working tree, so the README and `_assets/<name>.png` are written into it, while the Deploy to Pages step pushes the diagram to `gh-pages` (under `schemas/`) at once, which the README's image link points at.
- The workflow writes the diagram into `_assets/` without creating it, so the folder is made first.
- Its `uv add` steps change the schema's `pyproject.toml` and `uv.lock`, which are reverted, as they suit only a throwaway runner.
- The container runs as root, leaving the diagram root-owned and the schema's `.venv` pointing at the container's Python, which is fixed without `sudo` by `docker run --rm -v "$PWD/schemas/<name>":/w alpine sh -c 'rm -rf /w/.venv && chown -R 1000:1000 /w/_assets'` and then `uv sync` in the schema.

The generated README is committed as generated, in `docs(schemas): add <name> diagram` with the diagram, followed by `chore(schemas): bump <name> version` (`uv version --bump patch`), as `<name>-build-and-publish.yml` publishes to PyPI when a PR touching `schemas/<name>/**` merges, which needs a new version.
The `pyproject.toml` description follows the other schemas (`<X> extraction schema for MESA`) rather than the README sentence.
