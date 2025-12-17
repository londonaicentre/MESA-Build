import json
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from tests.llama import TestFineTuner


@pytest.fixture(scope="session")
def fine_tuner() -> TestFineTuner:
    return TestFineTuner("foo")


@patch("builtins.print")
@patch("builtins.open")
@patch("pathlib.Path.glob")
def test_generate_train_file(
    mock_glob: MagicMock,
    mock_file: MagicMock,
    mock_print: MagicMock,
    fine_tuner: TestFineTuner,
) -> None:
    samples_file_data: str = json.dumps({"content": "foo", "output": {"bar": "baz"}})
    file_write_mock: MagicMock = mock_open()
    file_read_mock: MagicMock = mock_open(read_data=samples_file_data)
    mock_file.side_effect = [file_write_mock(), file_read_mock()]
    mock_glob.return_value = ["sample001.json"]
    assert fine_tuner.generate_train_file("bar")
    assert mock_file.call_args_list == [
        call("train.jsonl", "w"),
        call("sample001.json"),
    ]
    assert (
        mock_print.call_args[0][0]
        == '{"instruction": "Extract the given information into a structured schema.", "context": "foo", "response": "{\\"bar\\": \\"baz\\"}"}'
    )


@patch("builtins.print")
@patch("builtins.open")
def test_generate_template_file(
    mock_file: MagicMock, mock_print: MagicMock, fine_tuner: TestFineTuner
) -> None:
    file_write_mock: MagicMock = mock_open()
    mock_file.return_value = file_write_mock()
    fine_tuner.generate_template_file("qux")
    mock_file.assert_called_with("template.json", "w")
    assert (
        mock_print.call_args[0][0]
        == '{"prompt": "qux\\n\\n### Instruction:\\n{instruction}\\n\\n### Input:\\n{context}\\n\\n", "completion": "{response}"}'
    )
