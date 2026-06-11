from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from conftest import BaseTrainerFactory, LoRATrainerFixture
from finetune.trainer import LoRATrainer
from fixtures import SchemaFixture


@pytest.fixture
def model_card() -> MagicMock:
    mock: MagicMock = MagicMock()
    mock.model_name = "foo"
    mock.major = 1
    mock.minor = 2
    mock.patch = 3
    mock.model_identifier = "foo_1_2_3"
    mock.to_yaml_bytes.return_value = b"bar"
    return mock


class TestMakeJobId:
    def test_make_job_id_format(
        self, mocker: MockerFixture, make_base_trainer: BaseTrainerFactory
    ) -> None:
        mock_datetime: MagicMock = mocker.patch("finetune.trainer.datetime")
        mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
        assert make_base_trainer().make_job_id("grault") == "20260101-120000-grault"


class TestBuildModelCard:
    def test_build_model_card_reads_state(
        self, make_base_trainer: BaseTrainerFactory
    ) -> None:
        card = make_base_trainer(
            model_name="foo", training_batch_names=["batch-a", "batch-b"]
        ).build_model_card(1, 2, 3)
        assert card.base_model_hf == "baz"
        assert card.model_name == "foo"
        assert card.major == 1
        assert card.minor == 2
        assert card.patch == 3
        assert card.model_description == "foo"
        assert card.training_data == ["batch-a", "batch-b"]
        assert card.output_schema is SchemaFixture

    def test_build_model_card_description_override(
        self, make_base_trainer: BaseTrainerFactory
    ) -> None:
        assert (
            make_base_trainer(description="foo")
            .build_model_card(1, 0, 0, model_description="a model")
            .model_description
            == "a model"
        )


class TestPrepareTrainingData:
    def test_prepare_training_data_delegates(
        self, mocker: MockerFixture, make_base_trainer: BaseTrainerFactory
    ) -> None:
        prepare: MagicMock = mocker.patch(
            "finetune.trainer.TrainingDataHandler.prepare", return_value="train.jsonl"
        )
        trainer: LoRATrainerFixture = make_base_trainer(
            training_batch_names=["batch-a"],
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "x"},
        )
        assert trainer.prepare_training_data("out.jsonl") == "train.jsonl"
        prepare.assert_called_once_with(
            schema=trainer.schema,
            system_prompt="foo",
            training_batch_names=["batch-a"],
            bucket="foo-bar",
            s3_prefix="trainingdata",
            output_file="out.jsonl",
            region="foo-bar-1",
            shuffle=True,
        )


class TestUploadModelFolder:
    def test_writes_model_card_and_uploads_each_file(
        self,
        tmp_path: Path,
        model_card: MagicMock,
        mocker: MockerFixture,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        upload: MagicMock = mocker.patch(
            "finetune.trainer.AWS.upload_file", return_value=True
        )
        target = tmp_path / "target"
        target.mkdir()
        (target / "model.safetensors").write_text("weights")
        (target / "config.json").write_text("{}")
        (target / "subdir").mkdir()

        make_base_trainer(
            aws_config={"bucket": "baz", "region": "qux", "role": "x"}
        ).upload_model_folder(str(target), model_card)

        assert (target / "model_card.yaml").read_bytes() == b"bar"
        uploaded = {call.kwargs["object_name"] for call in upload.call_args_list}
        assert uploaded == {"model.safetensors", "config.json", "model_card.yaml"}
        assert all(
            call.kwargs
            == {
                "region_name": "qux",
                "file_name": str(target / call.kwargs["object_name"]),
                "bucket": "baz",
                "object_name": call.kwargs["object_name"],
                "path": "models/foo/foo_1_2_3",
            }
            for call in upload.call_args_list
        )

    def test_upload_fails_raises_value_error(
        self,
        tmp_path: Path,
        model_card: MagicMock,
        mocker: MockerFixture,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        mocker.patch("finetune.trainer.AWS.upload_file", return_value=False)
        target = tmp_path / "target"
        target.mkdir()
        (target / "model.safetensors").write_text("weights")
        with pytest.raises(ValueError, match="Failed to upload"):
            make_base_trainer().upload_model_folder(str(target), model_card)


@dataclass
class ArchiveMocks:
    path: MagicMock
    tarfile: MagicMock
    tarinfo: MagicMock
    io: MagicMock
    aws: MagicMock


@pytest.fixture
def archive_mocks(mocker: MockerFixture) -> ArchiveMocks:
    return ArchiveMocks(
        mocker.patch("finetune.trainer.Path"),
        mocker.patch("finetune.trainer.tarfile.open"),
        mocker.patch("finetune.trainer.tarfile.TarInfo"),
        mocker.patch("finetune.trainer.io.BytesIO"),
        mocker.patch("finetune.trainer.AWS.upload_file"),
    )


class TestArchiveAndUpload:
    def test_archive_exists_calls_aws_upload_file(
        self,
        archive_mocks: ArchiveMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        archive_mocks.aws.return_value = True
        assert make_base_trainer(
            model_name="waldo",
            aws_config={"bucket": "x", "region": "qux", "role": "y"},
        ).archive_and_upload("grault/garply", model_card, "public-bucket")
        archive_mocks.tarfile.assert_not_called()
        archive_mocks.aws.assert_called_once_with(
            region_name="qux",
            file_name=str(
                archive_mocks.path.return_value.parent.__truediv__.return_value
            ),
            bucket="public-bucket",
            object_name="foo_1_2_3.tar.gz",
            path="waldo",
        )

    def test_default_bucket_is_public(
        self,
        archive_mocks: ArchiveMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        archive_mocks.aws.return_value = True
        make_base_trainer().archive_and_upload("bar/baz", model_card)
        assert (
            archive_mocks.aws.call_args.kwargs["bucket"]
            == "aicentre-nlpteam-mesa-public"
        )

    def test_archive_exists_upload_fails_raises_value_error(
        self,
        archive_mocks: ArchiveMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        archive_mocks.aws.return_value = False
        with pytest.raises(ValueError, match="Failed to upload merged model weights"):
            make_base_trainer().archive_and_upload("bar/baz", model_card)

    def test_archive_not_exists_creates_tarfile(
        self,
        archive_mocks: ArchiveMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.path.return_value.iterdir.return_value = []
        archive_mocks.aws.return_value = True
        make_base_trainer().archive_and_upload("bar/baz", model_card)
        archive_mocks.tarfile.assert_called_once_with(
            archive_mocks.path.return_value.parent.__truediv__.return_value, "w:gz"
        )

    def test_archive_not_exists_adds_all_target_items(
        self,
        archive_mocks: ArchiveMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.aws.return_value = True
        item1: MagicMock = MagicMock()
        item1.name = "baz.txt"
        item2: MagicMock = MagicMock()
        item2.name = "qux.txt"
        archive_mocks.path.return_value.iterdir.return_value = [item1, item2]
        make_base_trainer().archive_and_upload("quux/corge", model_card)
        mock_tar: MagicMock = archive_mocks.tarfile.return_value.__enter__.return_value
        assert mock_tar.add.call_count == 3
        mock_tar.add.assert_any_call(item1, arcname="baz.txt")
        mock_tar.add.assert_any_call(item2, arcname="qux.txt")

    def test_archive_not_exists_adds_model_card_yml(
        self,
        archive_mocks: ArchiveMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.path.return_value.iterdir.return_value = []
        archive_mocks.aws.return_value = True
        make_base_trainer().archive_and_upload("baz/qux", model_card)
        archive_mocks.tarinfo.assert_called_once_with(name="model_card.yml")
        archive_mocks.io.assert_called_once_with(b"bar")
        archive_mocks.tarfile.return_value.__enter__.return_value.addfile.assert_called_once_with(
            archive_mocks.tarinfo.return_value, archive_mocks.io.return_value
        )

    def test_archive_not_exists_adds_license(
        self,
        archive_mocks: ArchiveMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.path.return_value.iterdir.return_value = []
        archive_mocks.aws.return_value = True
        make_base_trainer().archive_and_upload("baz/qux", model_card)
        mock_tar: MagicMock = archive_mocks.tarfile.return_value.__enter__.return_value
        mock_tar.add.assert_called_once_with(
            archive_mocks.path.return_value.parents.__getitem__.return_value
            / "LICENSE.md",
            arcname="LICENSE.md",
        )

    def test_archive_not_exists_upload_fails_raises_value_error(
        self,
        archive_mocks: ArchiveMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.path.return_value.iterdir.return_value = []
        archive_mocks.aws.return_value = False
        with pytest.raises(ValueError, match="Failed to upload merged model weights"):
            make_base_trainer().archive_and_upload("baz/qux", model_card)


class TestPublish:
    def test_publish_uploads_folder(
        self,
        mocker: MockerFixture,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        upload: MagicMock = mocker.patch.object(LoRATrainer, "_upload_model_folder")
        mocker.patch.object(LoRATrainer, "_archive_and_upload")
        make_base_trainer().publish("target", model_card, False)
        upload.assert_called_once_with("target", model_card)

    def test_publish_default_does_not_archive(
        self,
        mocker: MockerFixture,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        mocker.patch.object(LoRATrainer, "_upload_model_folder")
        archive: MagicMock = mocker.patch.object(LoRATrainer, "_archive_and_upload")
        make_base_trainer().publish("target", model_card, False)
        archive.assert_not_called()

    def test_publish_push_public_archives(
        self,
        mocker: MockerFixture,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        mocker.patch.object(LoRATrainer, "_upload_model_folder")
        archive: MagicMock = mocker.patch.object(LoRATrainer, "_archive_and_upload")
        make_base_trainer().publish("target", model_card, True)
        archive.assert_called_once_with("target", model_card)
