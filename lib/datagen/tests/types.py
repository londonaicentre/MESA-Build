from dataclasses import dataclass
from unittest.mock import MagicMock


@dataclass
class PathOperations:
    read_text: MagicMock
    glob: MagicMock
    exists: MagicMock
