import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open

import pytest
from pytest_mock import MockerFixture

from finetune.trainingdata_handler import TrainingDataHandler
from fixtures import SchemaFixture
from mesa_types import TrainingMessage, TrainingSample


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
    exclude_overlong_samples: MagicMock


@dataclass
class ExcludeOverlongSamplesMocks:
    auto_tokenizer: MagicMock
    tokenizer: MagicMock


@pytest.fixture
def valid_sample_json() -> str:
    return json.dumps(
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


@pytest.fixture
def prepare_mocks(mocker: MockerFixture, valid_sample_json: str) -> PrepareMocks:
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
            mock_open(read_data=valid_sample_json + "\n"),
        ),
        mocker.patch("finetune.trainingdata_handler.random.shuffle"),
        mocker.patch("finetune.trainingdata_handler.logger"),
        mocker.patch(
            "finetune.trainingdata_handler.TrainingDataHandler.exclude_overlong_samples",
            side_effect=lambda samples, *args, **kwargs: samples,
        ),
    )


@pytest.fixture
def exclude_overlong_samples_mocks(
    mocker: MockerFixture,
) -> ExcludeOverlongSamplesMocks:
    mock_tokenizer: MagicMock = MagicMock()
    mock_auto_tokenizer: MagicMock = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    return ExcludeOverlongSamplesMocks(mock_auto_tokenizer, mock_tokenizer)


class TestS3Operations:
    # S3 discovery + download branches: correct prefix, download-when-uncached, and the
    # no-file / multiple-file / download-failure error paths. AWS/Path/open/shuffle mocked.
    def test_prepare_calls_list_s3_objects_with_correct_prefix(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(
            SchemaFixture,
            "foo",
            ["bar"],
            "foo.bar1.2baz",
            1024,
            bucket="baz",
            region="qux",
        )
        prepare_mocks.aws_mocks.list_s3_objects.assert_called_once_with(
            region_name="qux",
            bucket="baz",
            prefix="trainingdata/bar/",
        )

    def test_prepare_downloads_file_when_not_cached(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
        )
        prepare_mocks.aws_mocks.download_file.assert_called_once()

    def test_prepare_no_jsonl_file_raises_value_error(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.aws_mocks.list_s3_objects.return_value = []
        with pytest.raises(ValueError, match="No JSONL file found in foo"):
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
            )

    def test_prepare_multiple_jsonl_files_raises_value_error(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.aws_mocks.list_s3_objects.return_value = [
            {"Key": "trainingdata/foo/bar.jsonl"},
            {"Key": "trainingdata/foo/baz.jsonl"},
        ]
        with pytest.raises(ValueError, match="Multiple JSONL files found in foo"):
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
            )

    def test_prepare_download_fails_raises_value_error(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.aws_mocks.download_file.return_value = False
        with pytest.raises(ValueError, match="Failed to download"):
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
            )


class TestCaching:
    # Two distinct behaviours: create the cache dir, and skip the download when already cached.
    def test_prepare_creates_cache_directory(self, prepare_mocks: PrepareMocks) -> None:
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
        )
        prepare_mocks.path_mocks.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )

    def test_prepare_skips_download_when_cached(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.path_mocks.exists.return_value = True
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
        )
        prepare_mocks.aws_mocks.download_file.assert_not_called()


class TestSampleValidation:
    # Each invalid-sample path is skipped (with a logged warning) and an all-invalid batch raises:
    # system-prompt mismatch / schema mismatch / bad JSON / no-valid-samples. open/logger mocked.
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
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
            )
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
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
            )
        prepare_mocks.logger.warning.assert_called()

    def test_prepare_invalid_json_skips_sample(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.open_mock.return_value.__iter__ = lambda _: iter(["foo"])
        with pytest.raises(ValueError, match="No valid training samples found"):
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
            )
        prepare_mocks.logger.warning.assert_called()

    def test_prepare_no_valid_samples_raises_value_error(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        invalid_sample = {"messages": [{"role": "system", "content": "qux"}]}
        prepare_mocks.open_mock.return_value.__iter__ = lambda _: iter(
            [json.dumps(invalid_sample)]
        )
        with pytest.raises(ValueError, match="No valid training samples found"):
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
            )


class TestShuffle:
    # Distinct branches: shuffle on by default vs off when disabled. random.shuffle mocked.
    def test_prepare_shuffles_samples_by_default(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
        )
        prepare_mocks.shuffle.assert_called_once()

    def test_prepare_no_shuffle_when_disabled(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.shuffle.reset_mock()
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024, shuffle=False
        )
        prepare_mocks.shuffle.assert_not_called()


class TestOutput:
    def test_prepare_single_batch_valid_sample_returns_path(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        assert (
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
            )
            == "train.jsonl"
        )

    # prepare() echoes back the output_file it wrote to (a non-default value proves the behaviour).
    def test_prepare_custom_output_file_returns_custom_path(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        assert (
            TrainingDataHandler.prepare(
                SchemaFixture,
                "foo",
                ["foo"],
                "foo.bar1.2baz",
                1024,
                output_file="bar.jsonl",
            )
            == "bar.jsonl"
        )


class TestMultipleBatches:
    # Passing several batch names iterates S3 discovery once per batch. AWS.list_s3_objects mocked.
    def test_prepare_iterates_over_each_batch_name(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        prepare_mocks.aws_mocks.list_s3_objects.side_effect = [
            [{"Key": "trainingdata/foo/qux.jsonl"}],
            [{"Key": "trainingdata/bar/quux.jsonl"}],
        ]
        assert (
            TrainingDataHandler.prepare(
                SchemaFixture, "foo", ["foo", "bar"], "foo.bar1.2baz", 1024
            )
            == "train.jsonl"
        )
        assert prepare_mocks.aws_mocks.list_s3_objects.call_count == 2


class TestExcludeOverlongSamples:
    def test_prepare_calls_exclude_overlong_samples_with_valid_params(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
        )
        prepare_mocks.exclude_overlong_samples.assert_called_once()

    def test_prepare_skips_exclude_overlong_samples_with_empty_base_model(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"], "", 1024)
        prepare_mocks.exclude_overlong_samples.assert_not_called()

    def test_prepare_skips_exclude_overlong_samples_with_zero_max_seq_length(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 0)
        prepare_mocks.exclude_overlong_samples.assert_not_called()

    def test_prepare_warns_when_more_than_half_samples_excluded(
        self, prepare_mocks: PrepareMocks, valid_sample_json: str
    ) -> None:
        prepare_mocks.open_mock.return_value.__iter__ = lambda _: iter(
            [valid_sample_json] * 3
        )
        prepare_mocks.exclude_overlong_samples.side_effect = (
            lambda samples, *_, **__: samples[: len(samples) // 3]
        )
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
        )
        prepare_mocks.logger.warning.assert_called_once_with(
            "Samples consistently exceed max_seq_length, consider increasing"
        )

    @pytest.mark.parametrize(
        "total_samples,remaining_samples",
        [
            (3, 2),
            (2, 1),
        ],
    )
    def test_prepare_does_not_warn_when_half_or_fewer_samples_excluded(
        self,
        prepare_mocks: PrepareMocks,
        valid_sample_json: str,
        total_samples: int,
        remaining_samples: int,
    ) -> None:
        prepare_mocks.open_mock.return_value.__iter__ = lambda _: iter(
            [valid_sample_json] * total_samples
        )
        prepare_mocks.exclude_overlong_samples.side_effect = (
            lambda samples, *_, **__: samples[:remaining_samples]
        )
        TrainingDataHandler.prepare(
            SchemaFixture, "foo", ["foo"], "foo.bar1.2baz", 1024
        )
        prepare_mocks.logger.warning.assert_not_called()

    def test_prepare_does_not_warn_when_exclude_overlong_samples_not_called(
        self, prepare_mocks: PrepareMocks
    ) -> None:
        TrainingDataHandler.prepare(SchemaFixture, "foo", ["foo"], "", 1024)
        prepare_mocks.logger.warning.assert_not_called()

    def create_sample(self, content_length: int = 1) -> TrainingSample:
        return TrainingSample(
            messages=[
                TrainingMessage(role="system", content="a" * content_length),
                TrainingMessage(role="user", content="b" * content_length),
                TrainingMessage(role="assistant", content="c" * content_length),
            ]
        )

    def test_exclude_overlong_samples_loads_tokenizer_with_base_model(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        exclude_overlong_samples_mocks.tokenizer.encode.return_value = [1, 2, 3]
        TrainingDataHandler.exclude_overlong_samples(
            [self.create_sample()], 100, "foo.bar1.2baz"
        )
        exclude_overlong_samples_mocks.auto_tokenizer.assert_called_once_with(
            "foo.bar1.2baz", trust_remote_code=True
        )

    @pytest.mark.parametrize(
        "token_count,sample_count,expected_count",
        [
            (3, 1, 1),  # single sample under limit
            (200, 1, 0),  # single sample over limit
            (3, 2, 2),  # all samples under limit
            (200, 2, 0),  # all samples over limit
        ],
    )
    def test_exclude_overlong_samples_simple_cases(
        self,
        exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks,
        token_count: int,
        sample_count: int,
        expected_count: int,
    ) -> None:
        exclude_overlong_samples_mocks.tokenizer.encode.return_value = [1] * token_count
        assert (
            len(
                TrainingDataHandler.exclude_overlong_samples(
                    [self.create_sample() for _ in range(sample_count)], 100, "foo"
                )
            )
            == expected_count
        )

    def test_exclude_overlong_samples_mixed_lengths_returns_acceptable_samples(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        exclude_overlong_samples_mocks.tokenizer.encode.side_effect = lambda text: [
            1
        ] * len(text)
        result: list[TrainingSample] = TrainingDataHandler.exclude_overlong_samples(
            [
                self.create_sample(200),
                self.create_sample(150),
                self.create_sample(50),
                self.create_sample(30),
                self.create_sample(20),
                self.create_sample(10),
            ],
            100,
            "foo",
        )
        assert len(result) == 3
        assert all(
            sum(len(msg.content) for msg in sample.messages) < 100 for sample in result
        )

    def test_exclude_overlong_samples_respects_buffer_ratio(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        exclude_overlong_samples_mocks.tokenizer.encode.side_effect = lambda text: [
            1
        ] * len(text)
        assert (
            len(
                TrainingDataHandler.exclude_overlong_samples(
                    [self.create_sample(i * 3) for i in range(1, 11)],
                    50,
                    "foo",
                    buffer_ratio=0.2,
                )
            )
            == 5
        )

    def test_exclude_overlong_samples_sorts_by_total_character_length(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        exclude_overlong_samples_mocks.tokenizer.encode.return_value = [1, 2, 3]
        sample_short: TrainingSample = self.create_sample(10)
        sample_long: TrainingSample = self.create_sample(100)
        result: list[TrainingSample] = TrainingDataHandler.exclude_overlong_samples(
            [sample_short, sample_long], 1000, "foo"
        )
        assert result[0] == sample_long
        assert result[1] == sample_short

    def test_exclude_overlong_samples_empty_list_returns_empty(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        assert TrainingDataHandler.exclude_overlong_samples([], 100, "foo") == []

    def test_exclude_overlong_samples_includes_valid_samples_before_buffer_threshold(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        token_counts: list[int] = [150, 90, 40, 30]
        exclude_overlong_samples_mocks.tokenizer.encode.side_effect = lambda _: [
            1
        ] * token_counts.pop(0)
        assert (
            len(
                TrainingDataHandler.exclude_overlong_samples(
                    [
                        self.create_sample(100),
                        self.create_sample(80),
                        self.create_sample(40),
                        self.create_sample(30),
                    ],
                    100,
                    "foo",
                    buffer_ratio=0.5,
                )
            )
            == 3
        )

    def test_exclude_overlong_samples_includes_passing_sample_before_failed_sample(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        token_counts: list[int] = [50, 150, 40, 30, 20]
        exclude_overlong_samples_mocks.tokenizer.encode.side_effect = lambda _: [
            1
        ] * token_counts.pop(0)
        assert (
            len(
                TrainingDataHandler.exclude_overlong_samples(
                    [
                        self.create_sample(100),
                        self.create_sample(90),
                        self.create_sample(50),
                        self.create_sample(40),
                        self.create_sample(30),
                    ],
                    100,
                    "foo",
                    buffer_ratio=0.4,
                )
            )
            == 4
        )

    def test_exclude_overlong_samples_boundary_exactly_max_seq_length_fails(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        exclude_overlong_samples_mocks.tokenizer.encode.return_value = [1] * 100
        assert (
            TrainingDataHandler.exclude_overlong_samples(
                [self.create_sample()], 100, "foo"
            )
            == []
        )

    def test_exclude_overlong_samples_buffer_never_reached_returns_passing_samples(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        token_counts: list[int] = [50, 150, 40, 160]
        exclude_overlong_samples_mocks.tokenizer.encode.side_effect = lambda _: [
            1
        ] * token_counts.pop(0)
        assert (
            len(
                TrainingDataHandler.exclude_overlong_samples(
                    [
                        self.create_sample(100),
                        self.create_sample(90),
                        self.create_sample(50),
                        self.create_sample(40),
                    ],
                    100,
                    "foo",
                    buffer_ratio=0.5,
                )
            )
            == 2
        )

    def test_exclude_overlong_samples_buffer_size_equals_sample_count(
        self, exclude_overlong_samples_mocks: ExcludeOverlongSamplesMocks
    ) -> None:
        exclude_overlong_samples_mocks.tokenizer.encode.return_value = [1] * 50
        assert (
            len(
                TrainingDataHandler.exclude_overlong_samples(
                    [self.create_sample(), self.create_sample()],
                    100,
                    "foo",
                    buffer_ratio=1.0,
                )
            )
            == 2
        )
