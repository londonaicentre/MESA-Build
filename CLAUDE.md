# mesa-build

## Python versions

Every package must pin its Python version in a `.python-version` file.
Without one, CI's uv picks the newest CPython, and `uv sync` fails when a dependency has no wheel for it yet and needs a compiler the runner lacks (e.g. tiktoken, a litellm dependency, needs Rust on CPython 3.15).

Review this note once tiktoken publishes CPython 3.15 wheels (first seen 2026-10-20), as the pin may then only guard against the next Python release.

## PyPI trusted publishers

Each package's publish workflow (e.g. `conditionschema-build-and-publish.yml`) calls the reusable `build-and-publish.yml`, and PyPI matches a trusted publisher on the token's `job_workflow_ref`, which names the reusable workflow.
So a new package's pending trusted publisher names the workflow `build-and-publish.yml` (environment `pypi`), never the caller.
A pending publisher that names the caller is skipped, and the upload falls back to the active publisher the existing packages share, failing with `Non-user identities cannot create new projects`.
