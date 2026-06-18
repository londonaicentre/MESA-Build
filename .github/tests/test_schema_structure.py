import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from packaging.version import InvalidVersion, Version

SCHEMA_DIRS: list[Path] = sorted(
    d
    for d in (Path(__file__).parent.parent.parent / "schemas").iterdir()
    if d.is_dir() and not d.name.startswith("_")
)


@pytest.fixture(params=SCHEMA_DIRS, ids=[d.name for d in SCHEMA_DIRS])
def schema_dir(request: pytest.FixtureRequest) -> Path:
    return request.param


@pytest.fixture
def src(schema_dir: Path) -> Path:
    return schema_dir / "src" / schema_dir.name


@pytest.fixture
def top_level_names(schema_dir: Path) -> list[str]:
    return [f.name.lower() for f in schema_dir.iterdir() if f.is_file()]


class TestSourceStructure:
    REFERENCE_FILES: frozenset[str] = frozenset(
        [
            "__init__.py",
            "schema.py",
            "prompt_builder.py",
            "prompt_main.txt",
            "prompt_datagen.txt",
        ]
    )

    def test_src_package_dir_exists(self, src: Path) -> None:
        assert src.is_dir()

    def test_reference_files_present(self, src: Path) -> None:
        for filename in TestSourceStructure.REFERENCE_FILES:
            assert (src / filename).exists(), f"missing {filename}"

    def test_py_typed_marker_present(self, src: Path) -> None:
        assert (src / "py.typed").exists()

    def test_prompt_builder_extends_base(self, src: Path) -> None:
        content: str = (src / "prompt_builder.py").read_text()
        assert re.search(r"class \w+\(.*BasePromptBuilder.*\)", content)

    def test_init_exports_schema(self, schema_dir: Path) -> None:
        package: str = schema_dir.name
        sys.modules[f"{package}.schema"] = MagicMock()
        spec = importlib.util.spec_from_file_location(
            package,
            schema_dir / "src" / package / "__init__.py",
            submodule_search_locations=[str(schema_dir / "src" / package)],
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "Schema")
        finally:
            sys.modules.pop(package, None)
            sys.modules.pop(f"{package}.schema", None)


class TestPromptContent:
    def test_prompt_main_has_schema_placeholder(self, src: Path) -> None:
        assert re.search(r"\{SCHEMA\}", (src / "prompt_main.txt").read_text())

    def test_prompt_datagen_has_schema_placeholder(self, src: Path) -> None:
        assert re.search(r"\{SCHEMA\}", (src / "prompt_datagen.txt").read_text())

    def test_prompt_datagen_has_example_placeholder(self, src: Path) -> None:
        assert re.search(r"\{EXAMPLE\}", (src / "prompt_datagen.txt").read_text())


class TestTopLevelFiles:
    def test_tests_dir_exists(self, schema_dir: Path) -> None:
        assert (schema_dir / "tests").is_dir()

    def test_license_file_exists(self, top_level_names: list[str]) -> None:
        assert any(name.startswith("license.") for name in top_level_names)

    def test_readme_file_exists(self, top_level_names: list[str]) -> None:
        assert any(name.startswith("readme.") for name in top_level_names)


class TestPyproject:
    def _load(self, schema_dir: Path) -> dict:
        pyproject: Path = schema_dir / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml missing"
        with open(pyproject, "rb") as file:
            return tomllib.load(file)

    def test_pyproject_is_valid_toml(self, schema_dir: Path) -> None:
        self._load(schema_dir)

    def test_pyproject_has_project_section(self, schema_dir: Path) -> None:
        assert "project" in self._load(schema_dir)

    def test_pyproject_project_has_required_fields(self, schema_dir: Path) -> None:
        project: dict = self._load(schema_dir)["project"]
        for field in ("name", "version", "requires-python", "dependencies"):
            assert field in project, f"[project] missing '{field}'"

    def test_pyproject_has_build_system(self, schema_dir: Path) -> None:
        build: dict = self._load(schema_dir).get("build-system", {})
        assert "requires" in build and "build-backend" in build

    def test_pyproject_version_is_valid_pep440(self, schema_dir: Path) -> None:
        version: str = self._load(schema_dir)["project"]["version"]
        try:
            Version(version)
        except InvalidVersion:
            pytest.fail(f"'{version}' is not a valid PEP 440 version")


class TestExamples:
    def test_examples_dir_exists(self, src: Path) -> None:
        assert (src / "examples").is_dir()

    def test_examples_dir_contains_json(self, src: Path) -> None:
        assert list((src / "examples").glob("*.json")), (
            "no .json files found in examples/"
        )

    def test_example_json_files_are_valid(self, src: Path) -> None:
        for path in (src / "examples").glob("*.json"):
            try:
                json.loads(path.read_text())
            except json.JSONDecodeError as e:
                pytest.fail(f"{path.name} is not valid JSON: {e}")
