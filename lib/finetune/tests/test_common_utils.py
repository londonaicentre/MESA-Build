from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from finetune._common_utils import (
    archive_and_upload,
    build_model_card,
    make_job_id,
    upload_model_folder,
)
from fixtures import SchemaFixture


class TestMakeJobId:
    # Job id is "<timestamp>-<description>". datetime mocked for a deterministic stamp.
    def test_make_job_id_format(self, mocker: MockerFixture) -> None:
        mock_datetime: MagicMock = mocker.patch("finetune._common_utils.datetime")
        mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
        assert make_job_id("grault") == "20260101-120000-grault"


class TestBuildModelCard:
    # build_model_card forwards every constructor field through to the card unchanged.
    def test_build_model_card_forwards_fields(self) -> None:
        card = build_model_card(
            base_model="baz",
            model_name="foo",
            major=1,
            minor=2,
            patch=3,
            model_description="a model",
            training_data=["batch-a", "batch-b"],
            output_schema=SchemaFixture,
        )
        assert card.base_model_hf == "baz"
        assert card.model_name == "foo"
        assert card.major == 1
        assert card.minor == 2
        assert card.patch == 3
        assert card.model_description == "a model"
        assert card.training_data == ["batch-a", "batch-b"]
        assert card.output_schema is SchemaFixture


@pytest.fixture
def model_card() -> MagicMock:
    mock: MagicMock = MagicMock()
    mock.model_name = "foo"
    mock.major = 1
    mock.minor = 2
    mock.patch = 3
    mock.to_yaml_bytes.return_value = b"bar"
    return mock


class TestUploadModelFolder:
    # Writes model_card.yaml into the folder then uploads every file (skipping sub-dirs); an
    # upload failure raises. Uses a real tmp_path; only AWS.upload_file is mocked.
    def test_writes_model_card_and_uploads_each_file(
        self, tmp_path: Path, model_card: MagicMock, mocker: MockerFixture
    ) -> None:
        upload: MagicMock = mocker.patch(
            "finetune._common_utils.AWS.upload_file", return_value=True
        )
        (tmp_path / "model.safetensors").write_text("weights")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "subdir").mkdir()  # must be skipped

        upload_model_folder(
            target_folder=str(tmp_path),
            model_card=model_card,
            region="qux",
            bucket="baz",
        )

        # model_card.yaml written into the folder
        assert (tmp_path / "model_card.yaml").read_bytes() == b"bar"
        # one upload per file (model.safetensors, config.json, model_card.yaml); no sub-dir
        uploaded = {call.kwargs["object_name"] for call in upload.call_args_list}
        assert uploaded == {"model.safetensors", "config.json", "model_card.yaml"}
        assert all(
            call.kwargs == {
                "region_name": "qux",
                "file_name": str(tmp_path / call.kwargs["object_name"]),
                "bucket": "baz",
                "object_name": call.kwargs["object_name"],
                "path": "models/foo/foo_1_2_3",
            }
            for call in upload.call_args_list
        )

    def test_upload_fails_raises_value_error(
        self, tmp_path: Path, model_card: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch("finetune._common_utils.AWS.upload_file", return_value=False)
        (tmp_path / "model.safetensors").write_text("weights")
        with pytest.raises(ValueError, match="Failed to upload"):
            upload_model_folder(
                target_folder=str(tmp_path),
                model_card=model_card,
                region="qux",
                bucket="baz",
            )


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
        mocker.patch("finetune._common_utils.Path"),
        mocker.patch("finetune._common_utils.tarfile.open"),
        mocker.patch("finetune._common_utils.tarfile.TarInfo"),
        mocker.patch("finetune._common_utils.io.BytesIO"),
        mocker.patch("finetune._common_utils.AWS.upload_file"),
    )


class TestArchiveAndUpload:
    # Two arms: when the tarball already exists, skip creation and just upload (kwargs + returns
    # true); when it doesn't, build it from the target items + model_card.yml + LICENSE. The
    # default bucket is the public one, and an upload failure raises. Path/tarfile/io/AWS mocked.
    def test_archive_exists_calls_aws_upload_file(
        self, archive_mocks: ArchiveMocks, model_card: MagicMock
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        archive_mocks.aws.return_value = True
        assert archive_and_upload(
            target_folder="grault/garply",
            model_card=model_card,
            model_name="waldo",
            region="qux",
            bucket="public-bucket",
        )
        archive_mocks.tarfile.assert_not_called()  # existing tarball is reused, not rebuilt
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
        self, archive_mocks: ArchiveMocks, model_card: MagicMock
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        archive_mocks.aws.return_value = True
        archive_and_upload(
            target_folder="bar/baz", model_card=model_card, model_name="foo", region="qux"
        )
        assert (
            archive_mocks.aws.call_args.kwargs["bucket"]
            == "aicentre-nlpteam-mesa-public"
        )

    def test_archive_exists_upload_fails_raises_value_error(
        self, archive_mocks: ArchiveMocks, model_card: MagicMock
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = True
        archive_mocks.aws.return_value = False
        with pytest.raises(ValueError, match="Failed to upload merged model weights"):
            archive_and_upload(
                target_folder="bar/baz",
                model_card=model_card,
                model_name="foo",
                region="qux",
            )

    def test_archive_not_exists_creates_tarfile(
        self, archive_mocks: ArchiveMocks, model_card: MagicMock
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.path.return_value.iterdir.return_value = []
        archive_mocks.aws.return_value = True
        archive_and_upload(
            target_folder="bar/baz", model_card=model_card, model_name="foo", region="qux"
        )
        archive_mocks.tarfile.assert_called_once_with(
            archive_mocks.path.return_value.parent.__truediv__.return_value, "w:gz"
        )

    def test_archive_not_exists_adds_all_target_items(
        self, archive_mocks: ArchiveMocks, model_card: MagicMock
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.aws.return_value = True
        item1: MagicMock = MagicMock()
        item1.name = "baz.txt"
        item2: MagicMock = MagicMock()
        item2.name = "qux.txt"
        archive_mocks.path.return_value.iterdir.return_value = [item1, item2]
        archive_and_upload(
            target_folder="quux/corge",
            model_card=model_card,
            model_name="foo",
            region="qux",
        )
        mock_tar: MagicMock = archive_mocks.tarfile.return_value.__enter__.return_value
        # two target items + LICENSE.md
        assert mock_tar.add.call_count == 3
        mock_tar.add.assert_any_call(item1, arcname="baz.txt")
        mock_tar.add.assert_any_call(item2, arcname="qux.txt")

    def test_archive_not_exists_adds_model_card_yml(
        self, archive_mocks: ArchiveMocks, model_card: MagicMock
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.path.return_value.iterdir.return_value = []
        archive_mocks.aws.return_value = True
        archive_and_upload(
            target_folder="baz/qux", model_card=model_card, model_name="foo", region="qux"
        )
        archive_mocks.tarinfo.assert_called_once_with(name="model_card.yml")
        archive_mocks.io.assert_called_once_with(b"bar")
        archive_mocks.tarfile.return_value.__enter__.return_value.addfile.assert_called_once_with(
            archive_mocks.tarinfo.return_value, archive_mocks.io.return_value
        )

    def test_archive_not_exists_adds_license(
        self, archive_mocks: ArchiveMocks, model_card: MagicMock
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.path.return_value.iterdir.return_value = []
        archive_mocks.aws.return_value = True
        archive_and_upload(
            target_folder="baz/qux", model_card=model_card, model_name="foo", region="qux"
        )
        mock_tar: MagicMock = archive_mocks.tarfile.return_value.__enter__.return_value
        mock_tar.add.assert_called_once_with(
            archive_mocks.path.return_value.parents.__getitem__.return_value
            / "LICENSE.md",
            arcname="LICENSE.md",
        )

    def test_archive_not_exists_upload_fails_raises_value_error(
        self, archive_mocks: ArchiveMocks, model_card: MagicMock
    ) -> None:
        archive_mocks.path.return_value.parent.__truediv__.return_value.exists.return_value = False
        archive_mocks.path.return_value.iterdir.return_value = []
        archive_mocks.aws.return_value = False
        with pytest.raises(ValueError, match="Failed to upload merged model weights"):
            archive_and_upload(
                target_folder="baz/qux",
                model_card=model_card,
                model_name="foo",
                region="qux",
            )
