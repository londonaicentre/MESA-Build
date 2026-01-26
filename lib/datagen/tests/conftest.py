from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from tests.types import PathOperations


@pytest.fixture
def mock_path_operations(mocker: MockerFixture) -> PathOperations:
    mock_glob: MagicMock = mocker.patch.object(Path, "glob")
    mock_glob.return_value = sorted(
        Path(f"./data/documents/foo/document_{index}.json") for index in range(5)
    )
    return PathOperations(
        read_text=mocker.patch.object(
            Path,
            "read_text",
            return_value='{"source": "foo", "content": "bar", "timestamp": "2026-01-01T00:00:00Z"}',
        ),
        glob=mock_glob,
        exists=mocker.patch.object(Path, "exists", return_value=True),
    )
