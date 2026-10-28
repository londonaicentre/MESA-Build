import json
from pathlib import Path

import pytest

from datagen.version_detector import get_schema_version


@pytest.fixture
def site_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(tmp_path)
    return tmp_path


class TestGetSchemaVersion:
    @staticmethod
    def install(
        site_packages: Path,
        version: str,
        direct_url: dict[str, object] | None = None,
    ) -> None:
        dist_info: Path = site_packages / f"londonaicentre_foo-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: londonaicentre-foo\nVersion: {version}\n"
        )
        if direct_url is not None:
            (dist_info / "direct_url.json").write_text(json.dumps(direct_url))

    @staticmethod
    def create_source(site_packages: Path, version: str) -> Path:
        source: Path = site_packages / "source"
        source.mkdir()
        (source / "pyproject.toml").write_text(
            f'[project]\nname = "londonaicentre-foo"\nversion = "{version}"\n'
        )
        return source

    @pytest.mark.parametrize(
        "version,expected",
        [
            ("1.2.3", "1_2_3"),
            ("1.2.3.4.5", "1_2_3"),
        ],
    )
    def test_get_schema_version_installed_package_returns_formatted_version(
        self, site_packages: Path, version: str, expected: str
    ) -> None:
        TestGetSchemaVersion.install(site_packages, version)
        assert get_schema_version("foo") == expected

    def test_get_schema_version_editable_package_returns_source_version(
        self, site_packages: Path
    ) -> None:
        TestGetSchemaVersion.install(
            site_packages,
            "1.2.3",
            {
                "url": TestGetSchemaVersion.create_source(
                    site_packages, "1.3.0"
                ).as_uri(),
                "dir_info": {"editable": True},
            },
        )
        assert get_schema_version("foo") == "1_3_0"

    def test_get_schema_version_non_editable_direct_url_returns_metadata_version(
        self, site_packages: Path
    ) -> None:
        TestGetSchemaVersion.install(
            site_packages,
            "1.2.3",
            {
                "url": TestGetSchemaVersion.create_source(
                    site_packages, "1.3.0"
                ).as_uri(),
                "dir_info": {},
            },
        )
        assert get_schema_version("foo") == "1_2_3"

    def test_get_schema_version_package_not_found_raises_runtime_error(
        self, site_packages: Path
    ) -> None:
        with pytest.raises(
            RuntimeError, match="Schema package 'londonaicentre-baz' not found"
        ):
            get_schema_version("baz")
