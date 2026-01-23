"""
version_detector.py

Schema version detection
"""

from importlib.metadata import PackageNotFoundError, version


def get_schema_version(schema_name: str) -> str:
    """
    Get version string for schema package
    """
    pypi_package = f"londonaicentre-{schema_name}"
    try:
        raw_version = version(pypi_package)
        version_parts = raw_version.split(".")[:3]
        return "_".join(version_parts)
    except PackageNotFoundError as e:
        raise RuntimeError(f"Schema package '{pypi_package}' not found") from e
