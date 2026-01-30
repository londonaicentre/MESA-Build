from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from datagen.version_detector import get_schema_version


@dataclass
class VersionMocks:
    version: MagicMock


@pytest.fixture
def mock_version(mocker: MockerFixture) -> VersionMocks:
    return VersionMocks(
        version=mocker.patch("datagen.version_detector.version", autospec=True)
    )


def test_get_schema_version_valid_package_returns_formatted_version(
    mock_version: VersionMocks,
) -> None:
    mock_version.version.return_value = "1.2.3"
    assert get_schema_version("foo") == "1_2_3"
    mock_version.version.assert_called_once_with("londonaicentre-foo")


def test_get_schema_version_version_with_extra_parts_returns_first_three(
    mock_version: VersionMocks,
) -> None:
    mock_version.version.return_value = "1.2.3.4.5"
    assert get_schema_version("bar") == "1_2_3"


def test_get_schema_version_package_not_found_raises_runtime_error(
    mock_version: VersionMocks,
) -> None:
    mock_version.version.side_effect = PackageNotFoundError("foo")
    with pytest.raises(
        RuntimeError, match="Schema package 'londonaicentre-baz' not found"
    ):
        get_schema_version("baz")
