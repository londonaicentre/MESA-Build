"""
version_detector.py

Schema version detection
"""

import json
import tomllib
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname


def get_schema_version(schema_name: str) -> str:
    """
    Get the version string of a schema package, read from source if editable.
    """
    pypi_package = f"londonaicentre-{schema_name}"
    try:
        package: Distribution = distribution(pypi_package)
    except PackageNotFoundError as e:
        raise RuntimeError(f"Schema package '{pypi_package}' not found") from e
    direct_url: dict[str, Any] = json.loads(
        package.read_text("direct_url.json") or "{}"
    )
    raw_version: str = (
        tomllib.loads(
            (
                Path(url2pathname(urlparse(direct_url["url"]).path)) / "pyproject.toml"
            ).read_text()
        )["project"]["version"]
        if direct_url.get("dir_info", {}).get("editable")
        else package.version
    )
    return "_".join(raw_version.split(".")[:3])
