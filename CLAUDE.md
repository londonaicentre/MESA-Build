# mesa-build

## Python versions

Every package must pin its Python version in a `.python-version` file.
Without one, CI's uv picks the newest CPython, and `uv sync` fails when a dependency has no wheel for it yet and needs a compiler the runner lacks (e.g. tiktoken, a litellm dependency, needs Rust on CPython 3.15).

Review this note once tiktoken publishes CPython 3.15 wheels (first seen 2026-10-20), as the pin may then only guard against the next Python release.
