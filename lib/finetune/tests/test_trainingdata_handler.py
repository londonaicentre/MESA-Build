import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open

import pytest
from pytest_mock import MockerFixture

from finetune.trainingdata_handler import TrainingDataHandler
from fixtures import SchemaFixture


@dataclass
class AWSMocks:
    list_s3_objects: MagicMock
    download_file: MagicMock


@dataclass
class PathMocks:
    mkdir: MagicMock
    exists: MagicMock


@dataclass
class PrepareMocks:
    path_mocks: PathMocks
    aws_mocks: AWSMocks
    open_mock: MagicMock
    shuffle: MagicMock
    logger: MagicMock


@pytest.fixture
def prepare_mocks(mocker: MockerFixture) -> PrepareMocks:
    path_mocks: PathMocks = PathMocks(
        mocker.patch.object(Path, "mkdir"),
        mocker.patch.object(Path, "exists", return_value=False),
    )
    aws_mocks: AWSMocks = AWSMocks(
        mocker.patch(
            "finetune.trainingdata_handler.AWS.list_s3_objects",
            return_value=[{"Key": "trainingdata/foo/bar.jsonl"}],
        ),
        mocker.patch(
            "finetune.trainingdata_handler.AWS.download_file", return_value=True
        ),
    )
    return PrepareMocks(
        path_mocks,
        aws_mocks,
        mocker.patch(
            "builtins.open",
            mock_open(
                read_data=json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "foo"},
                            {"role": "user", "content": "bar"},
                            {
                                "role": "assistant",
                                "content": '<output>{"foo": "baz", "bar": 42}</output>',
                            },
                        ]
                    }
                )
                + "\n"
            ),
        ),
        mocker.patch("finetune.trainingdata_handler.random.shuffle"),
        mocker.patch("finetune.trainingdata_handler.logger"),
    )


class TestS3Operations:
    def test_prepare_calls_list_s3_objects_with_correct_prefix(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["bar"], bucket="baz", region="qux"
        )
        prepare_mocks.aws_mocks.list_s3_objects.assert_called_once_with(
            region_name="qux",
            bucket="baz",
            prefix="trainingdata/bar/",
        )

    def test_prepare_downloads_file_when_not_cached(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])
        prepare_mocks.aws_mocks.download_file.assert_called_once()

    def test_prepare_no_jsonl_file_raises_value_error(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.aws_mocks.list_s3_objects.return_value = []
        with pytest.raises(ValueError, match="No JSONL file found in foo"):
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])

    def test_prepare_multiple_jsonl_files_raises_value_error(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.aws_mocks.list_s3_objects.return_value = [
            {"Key": "trainingdata/foo/bar.jsonl"},
            {"Key": "trainingdata/foo/baz.jsonl"},
        ]
        with pytest.raises(ValueError, match="Multiple JSONL files found in foo"):
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])

    def test_prepare_download_fails_raises_value_error(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.aws_mocks.download_file.return_value = False
        with pytest.raises(ValueError, match="Failed to download"):
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])


class TestCaching:
    def test_prepare_creates_cache_directory(self, prepare_mocks: PrepareMocks) -> None:
        TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])
        prepare_mocks.path_mocks.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )

    def test_prepare_skips_download_when_cached(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.path_mocks.exists.return_value = True
        TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])
        prepare_mocks.aws_mocks.download_file.assert_not_called()


class TestSampleValidation:
    def test_prepare_system_prompt_mismatch_skips_sample(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        mismatched_sample: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "qux"},
                {"role": "user", "content": "quux"},
                {
                    "role": "assistant",
                    "content": '<output>{"foo": "corge", "bar": 42}</output>',
                },
            ]
        }
        prepare_mocks.open_mock.return_value.read.return_value = (
            json.dumps(mismatched_sample) + "\n"
        )
        prepare_mocks.open_mock.return_value.__iter__ = lambda _: iter(
            [json.dumps(mismatched_sample)]
        )
        with pytest.raises(ValueError, match="No valid training samples found"):
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])
        prepare_mocks.logger.warning.assert_called()

    def test_prepare_invalid_schema_skips_sample(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        invalid_schema_sample: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "foo"},
                {"role": "user", "content": "bar"},
                {
                    "role": "assistant",
                    "content": '<output>{"qux": "quux"}</output>',
                },
            ]
        }
        prepare_mocks.open_mock.return_value.__iter__ = lambda _: iter(
            [json.dumps(invalid_schema_sample)]
        )
        with pytest.raises(ValueError, match="No valid training samples found"):
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])
        prepare_mocks.logger.warning.assert_called()

    def test_prepare_invalid_json_skips_sample(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.open_mock.return_value.__iter__ = lambda _: iter(["foo"])
        with pytest.raises(ValueError, match="No valid training samples found"):
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])
        prepare_mocks.logger.warning.assert_called()

    def test_prepare_no_valid_samples_raises_value_error(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        invalid_sample = {"messages": [{"role": "system", "content": "qux"}]}
        prepare_mocks.open_mock.return_value.__iter__ = lambda _: iter(
            [json.dumps(invalid_sample)]
        )
        with pytest.raises(ValueError, match="No valid training samples found"):
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])


class TestShuffle:
    def test_prepare_shuffles_samples_by_default(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"])
        prepare_mocks.shuffle.assert_called_once()

    def test_prepare_no_shuffle_when_disabled(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.shuffle.reset_mock()
        TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"], shuffle=False)
        prepare_mocks.shuffle.assert_not_called()


class TestOutput:
    def test_prepare_single_batch_valid_sample_returns_path(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        assert (
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"]) == "train.jsonl"
        )

    def test_prepare_custom_output_file_returns_custom_path(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        assert (
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], output_file="bar.jsonl"
            )
            == "bar.jsonl"
        )


class TestMultipleBatches:
    def test_prepare_multiple_batches_combines_samples(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.aws_mocks.list_s3_objects.side_effect = [
            [{"Key": "trainingdata/foo/qux.jsonl"}],
            [{"Key": "trainingdata/bar/quux.jsonl"}],
        ]
        assert (
            TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo", "bar"])
            == "train.jsonl"
        )
        assert prepare_mocks.aws_mocks.list_s3_objects.call_count == 2
