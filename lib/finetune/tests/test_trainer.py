import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from conftest import BaseTrainerFactory, LoRATrainerFixture, PromptBuilderFixture
from finetune import trainer
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
    @pytest.mark.parametrize(
        "description, expected",
        [
            ("grault", "20260101-120000-grault"),
            ("MLX retrain - spike", "20260101-120000-mlx-retrain-spike"),
            ("***", "20260101-120000"),
            ("a" * 60, f"20260101-120000-{'a' * 40}"),
        ],
    )
    def test_make_job_id_slugifies_description(
        self,
        mocker: MockerFixture,
        make_base_trainer: BaseTrainerFactory,
        description: str,
        expected: str,
    ) -> None:
        mock_datetime: MagicMock = mocker.patch("finetune.trainer.datetime")
        mock_datetime.now.return_value.strftime.return_value = "20260101-120000"
        assert make_base_trainer().make_job_id(description) == expected


@dataclass
class SchemaMetadataMocks:
    packages_distributions: MagicMock
    version: MagicMock
    load_prompt_parameters: MagicMock


@pytest.fixture
def schema_metadata_mocks(mocker: MockerFixture) -> SchemaMetadataMocks:
    return SchemaMetadataMocks(
        mocker.patch("importlib.metadata.packages_distributions", return_value={}),
        mocker.patch("importlib.metadata.version", return_value="1.0.0"),
        mocker.patch(
            "finetune.trainer.TrainingDataHandler.load_prompt_parameters",
            return_value=None,
        ),
    )


class TestBuildModelCard:
    def test_build_model_card_reads_state(
        self,
        make_base_trainer: BaseTrainerFactory,
        schema_metadata_mocks: SchemaMetadataMocks,
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
        assert card.prompt_parameters is None

    def test_build_model_card_parameterised_batches_records_parameters(
        self,
        make_base_trainer: BaseTrainerFactory,
        schema_metadata_mocks: SchemaMetadataMocks,
    ) -> None:
        schema_metadata_mocks.load_prompt_parameters.side_effect = [
            {"condition": "foo"},
            None,
        ]
        assert make_base_trainer(
            training_batch_names=["batch-a", "batch-b"]
        ).build_model_card(1, 2, 3).prompt_parameters == [{"condition": "foo"}]
        schema_metadata_mocks.load_prompt_parameters.assert_any_call(
            "batch-a", "qux", "trainingdata", "quux"
        )

    def test_build_model_card_description_override(
        self,
        make_base_trainer: BaseTrainerFactory,
        schema_metadata_mocks: SchemaMetadataMocks,
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
            prompt_builder=trainer.prompt_builder,
            training_batch_names=["batch-a"],
            base_model="baz",
            max_seq_length=2048,
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

        assert (target / "model_card.yml").read_bytes() == b"bar"
        uploaded = {call.kwargs["object_name"] for call in upload.call_args_list}
        assert uploaded == {"model.safetensors", "config.json", "model_card.yml"}
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
class ArchiveAndUploadMocks:
    _add_model_card_and_license: MagicMock
    _upload_archive: MagicMock


@pytest.fixture
def archive_and_upload_mocks(mocker: MockerFixture) -> ArchiveAndUploadMocks:
    return ArchiveAndUploadMocks(
        mocker.patch.object(LoRATrainer, "_add_model_card_and_license"),
        mocker.patch.object(LoRATrainer, "_upload_archive"),
    )


@dataclass
class MakePublicMocks:
    list_s3_objects: MagicMock
    _download: MagicMock
    _archive_and_upload: MagicMock


@pytest.fixture
def make_public_mocks(mocker: MockerFixture) -> MakePublicMocks:
    return MakePublicMocks(
        mocker.patch(
            "finetune.trainer.AWS.list_s3_objects",
            return_value=[
                {"Key": "models/foo/foo_1_2_3/"},
                {"Key": "models/foo/foo_1_2_3/bar.safetensors"},
                {"Key": "models/foo/foo_1_2_3/model_card.yaml"},
            ],
        ),
        mocker.patch.object(LoRATrainer, "_download"),
        mocker.patch.object(LoRATrainer, "_archive_and_upload"),
    )


@dataclass
class RepackageModelMocks:
    _download: MagicMock
    _add_model_card_and_license: MagicMock
    _upload_archive: MagicMock


@pytest.fixture
def repackage_model_mocks(mocker: MockerFixture) -> RepackageModelMocks:
    return RepackageModelMocks(
        mocker.patch.object(LoRATrainer, "_download"),
        mocker.patch.object(LoRATrainer, "_add_model_card_and_license"),
        mocker.patch.object(LoRATrainer, "_upload_archive"),
    )


@pytest.fixture
def license_text() -> bytes:
    return (Path(trainer.__file__).parents[2] / "LICENSE.md").read_bytes()


def write_tar(file: Path | io.BytesIO, members: dict[str, bytes]) -> None:
    with (
        tarfile.open(fileobj=file, mode="w:gz")
        if isinstance(file, io.BytesIO)
        else tarfile.open(file, "w:gz") as tar
    ):
        for name, content in members.items():
            tarinfo: tarfile.TarInfo = tarfile.TarInfo(name)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content))


def read_tar(file: Path | io.BytesIO) -> list[tuple[str, bytes]]:
    with (
        tarfile.open(fileobj=file, mode="r:gz")
        if isinstance(file, io.BytesIO)
        else tarfile.open(file, "r:gz") as tar
    ):
        return [
            (member.name, (tar.extractfile(member) or io.BytesIO()).read())
            for member in tar
        ]


class TestDownload:
    def test_download_creates_folder_and_returns_path(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        download: MagicMock = mocker.patch(
            "finetune.trainer.AWS.download_file", return_value=True
        )
        assert LoRATrainerFixture.download(
            "qux", "bar", tmp_path / "baz" / "foo.tar.gz", "quux"
        ) == (tmp_path / "baz" / "foo.tar.gz")
        assert (tmp_path / "baz").is_dir()
        download.assert_called_once_with(
            "qux", "bar", str(tmp_path / "baz" / "foo.tar.gz"), "foo.tar.gz", "quux"
        )

    def test_download_download_fails_raises_value_error(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        mocker.patch("finetune.trainer.AWS.download_file", return_value=False)
        with pytest.raises(ValueError, match="Failed to download quux/foo.tar.gz"):
            LoRATrainerFixture.download("qux", "bar", tmp_path / "foo.tar.gz", "quux")


class TestAddModelCardAndLicense:
    def test_add_model_card_and_license_appends_card_then_license(
        self, license_text: bytes
    ) -> None:
        file: io.BytesIO = io.BytesIO()
        with tarfile.open(fileobj=file, mode="w:gz") as tar:
            LoRATrainerFixture.add_model_card_and_license(tar, b"baz")
        file.seek(0)
        assert read_tar(file) == [
            ("model_card.yml", b"baz"),
            ("LICENSE.md", license_text),
        ]


class TestUploadArchive:
    def test_upload_archive_calls_aws_upload_file(self, mocker: MockerFixture) -> None:
        upload: MagicMock = mocker.patch(
            "finetune.trainer.AWS.upload_file", return_value=True
        )
        LoRATrainerFixture.upload_archive(
            "qux", Path("grault/foo_1_2_3.tar.gz"), "bar", "waldo", True
        )
        upload.assert_called_once_with(
            region_name="qux",
            file_name="grault/foo_1_2_3.tar.gz",
            bucket="bar",
            object_name="foo_1_2_3.tar.gz",
            path="waldo",
            force=True,
        )

    def test_upload_archive_upload_fails_raises_value_error(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("finetune.trainer.AWS.upload_file", return_value=False)
        with pytest.raises(ValueError, match="Failed to upload merged model weights"):
            LoRATrainerFixture.upload_archive(
                "qux", Path("foo_1_2_3.tar.gz"), "bar", "waldo", False
            )


class TestArchiveAndUpload:
    @staticmethod
    def make_target(tmp_path: Path) -> Path:
        (tmp_path / "foo_1_2_3").mkdir()
        (tmp_path / "foo_1_2_3" / "bar.safetensors").write_bytes(b"baz")
        (tmp_path / "foo_1_2_3" / "model_card.yml").write_bytes(b"qux")
        (tmp_path / "foo_1_2_3" / "model_card.yaml").write_bytes(b"quux")
        return tmp_path / "foo_1_2_3"

    @pytest.mark.parametrize(
        "exists, force", [(False, False), (False, True), (True, True)]
    )
    def test_archive_and_upload_archive_missing_or_forced_writes_items(
        self,
        tmp_path: Path,
        archive_and_upload_mocks: ArchiveAndUploadMocks,
        exists: bool,
        force: bool,
    ) -> None:
        if exists:
            write_tar(tmp_path / "foo_1_2_3.tar.gz", {"corge": b"grault"})
        LoRATrainerFixture.archive_and_upload(
            "qux",
            TestArchiveAndUpload.make_target(tmp_path),
            "waldo",
            "foo_1_2_3",
            b"bar",
            "garply",
            force,
        )
        assert read_tar(tmp_path / "foo_1_2_3.tar.gz") == [("bar.safetensors", b"baz")]
        assert (
            archive_and_upload_mocks._add_model_card_and_license.call_args.args[1]
            == b"bar"
        )

    def test_archive_and_upload_archive_exists_not_forced_keeps_archive(
        self, tmp_path: Path, archive_and_upload_mocks: ArchiveAndUploadMocks
    ) -> None:
        write_tar(tmp_path / "foo_1_2_3.tar.gz", {"corge": b"grault"})
        LoRATrainerFixture.archive_and_upload(
            "qux",
            TestArchiveAndUpload.make_target(tmp_path),
            "waldo",
            "foo_1_2_3",
            b"bar",
            "garply",
            False,
        )
        assert read_tar(tmp_path / "foo_1_2_3.tar.gz") == [("corge", b"grault")]
        archive_and_upload_mocks._add_model_card_and_license.assert_not_called()

    @pytest.mark.parametrize("force", [True, False])
    def test_archive_and_upload_uploads_archive(
        self,
        tmp_path: Path,
        archive_and_upload_mocks: ArchiveAndUploadMocks,
        force: bool,
    ) -> None:
        LoRATrainerFixture.archive_and_upload(
            "qux",
            TestArchiveAndUpload.make_target(tmp_path),
            "waldo",
            "foo_1_2_3",
            b"bar",
            "garply",
            force,
        )
        archive_and_upload_mocks._upload_archive.assert_called_once_with(
            "qux", tmp_path / "foo_1_2_3.tar.gz", "garply", "waldo", force
        )


class TestMakePublic:
    def test_make_public_downloads_each_model_file(
        self, tmp_path: Path, make_public_mocks: MakePublicMocks
    ) -> None:
        (tmp_path / "foo_1_2_3").mkdir()
        (tmp_path / "foo_1_2_3" / "model_card.yaml").write_bytes(b"bar")
        LoRATrainer.make_public("foo", "foo_1_2_3", str(tmp_path), "qux", "baz")
        make_public_mocks.list_s3_objects.assert_called_once_with(
            "qux", "baz", "models/foo/foo_1_2_3/"
        )
        assert [
            download_call.args
            for download_call in make_public_mocks._download.call_args_list
        ] == [
            ("qux", "baz", tmp_path / "foo_1_2_3" / file_name, "models/foo/foo_1_2_3")
            for file_name in ["bar.safetensors", "model_card.yaml"]
        ]

    @pytest.mark.parametrize(
        "model_cards, model_card_bytes",
        [
            ({"model_card.yml": b"bar"}, b"bar"),
            ({"model_card.yaml": b"baz"}, b"baz"),
            ({"model_card.yml": b"bar", "model_card.yaml": b"baz"}, b"bar"),
        ],
    )
    def test_make_public_archives_and_uploads_with_force(
        self,
        tmp_path: Path,
        make_public_mocks: MakePublicMocks,
        model_cards: dict[str, bytes],
        model_card_bytes: bytes,
    ) -> None:
        (tmp_path / "foo_1_2_3").mkdir()
        for name, content in model_cards.items():
            (tmp_path / "foo_1_2_3" / name).write_bytes(content)
        LoRATrainer.make_public("foo", "foo_1_2_3", str(tmp_path))
        make_public_mocks._archive_and_upload.assert_called_once_with(
            "eu-west-2",
            tmp_path / "foo_1_2_3",
            "foo",
            "foo_1_2_3",
            model_card_bytes,
            "aicentre-nlpteam-mesa-public",
            True,
        )

    def test_make_public_no_model_card_raises_value_error(
        self, tmp_path: Path, make_public_mocks: MakePublicMocks
    ) -> None:
        with pytest.raises(ValueError, match="models/foo/foo_1_2_3 has no model card"):
            LoRATrainer.make_public("foo", "foo_1_2_3", str(tmp_path))
        make_public_mocks._archive_and_upload.assert_not_called()


class TestRepackageModel:
    @staticmethod
    def write_source(
        repackage_model_mocks: RepackageModelMocks, members: dict[str, bytes]
    ) -> None:
        def download(_: str, __: str, file_path: Path, ___: str) -> Path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            write_tar(file_path, members)
            return file_path

        repackage_model_mocks._download.side_effect = download

    @pytest.mark.parametrize(
        "source, model_card_bytes",
        [
            ({"bar.safetensors": b"baz", "model_card.yml": b"qux"}, b"qux"),
            ({"bar.safetensors": b"baz", "./model_card.yml": b"qux"}, b"qux"),
            (
                {
                    "bar.safetensors": b"baz",
                    "model_card.yaml": b"quux",
                    "model_card.yml": b"qux",
                },
                b"qux",
            ),
        ],
    )
    def test_repackage_model_existing_model_card_yml_is_kept(
        self,
        tmp_path: Path,
        repackage_model_mocks: RepackageModelMocks,
        source: dict[str, bytes],
        model_card_bytes: bytes,
    ) -> None:
        TestRepackageModel.write_source(repackage_model_mocks, source)
        LoRATrainer.repackage_model("foo", "foo_1_2_3", str(tmp_path))
        assert (
            repackage_model_mocks._add_model_card_and_license.call_args.args[1]
            == model_card_bytes
        )

    def test_repackage_model_model_card_yaml_only_is_renamed(
        self, tmp_path: Path, repackage_model_mocks: RepackageModelMocks
    ) -> None:
        TestRepackageModel.write_source(
            repackage_model_mocks,
            {"bar.safetensors": b"baz", "model_card.yaml": b"quux"},
        )
        LoRATrainer.repackage_model("foo", "foo_1_2_3", str(tmp_path))
        assert (
            repackage_model_mocks._add_model_card_and_license.call_args.args[1]
            == b"quux"
        )
        assert read_tar(tmp_path / "foo_1_2_3.tar.gz") == [("bar.safetensors", b"baz")]

    def test_repackage_model_given_model_card_replaces_existing(
        self,
        tmp_path: Path,
        repackage_model_mocks: RepackageModelMocks,
        model_card: MagicMock,
    ) -> None:
        TestRepackageModel.write_source(
            repackage_model_mocks,
            {"bar.safetensors": b"baz", "model_card.yml": b"qux"},
        )
        LoRATrainer.repackage_model("foo", "foo_1_2_3", str(tmp_path), model_card)
        assert (
            repackage_model_mocks._add_model_card_and_license.call_args.args[1]
            == b"bar"
        )

    @pytest.mark.parametrize("license_name", ["LICENSE.md", "./LICENSE.md", None])
    def test_repackage_model_drops_existing_license_and_card(
        self,
        tmp_path: Path,
        repackage_model_mocks: RepackageModelMocks,
        license_name: str | None,
    ) -> None:
        TestRepackageModel.write_source(
            repackage_model_mocks,
            {
                "bar.safetensors": b"baz",
                "model_card.yml": b"qux",
                "model_card.yaml": b"quux",
            }
            | ({} if license_name is None else {license_name: b"corge"}),
        )
        LoRATrainer.repackage_model("foo", "foo_1_2_3", str(tmp_path))
        assert read_tar(tmp_path / "foo_1_2_3.tar.gz") == [("bar.safetensors", b"baz")]

    def test_repackage_model_uploads_archive_with_force(
        self, tmp_path: Path, repackage_model_mocks: RepackageModelMocks
    ) -> None:
        TestRepackageModel.write_source(
            repackage_model_mocks, {"model_card.yml": b"qux"}
        )
        LoRATrainer.repackage_model("foo", "foo_1_2_3", str(tmp_path))
        repackage_model_mocks._upload_archive.assert_called_once_with(
            "eu-west-2",
            tmp_path / "foo_1_2_3.tar.gz",
            "aicentre-nlpteam-mesa-public",
            "foo",
            True,
        )

    def test_repackage_model_no_model_card_raises_value_error(
        self, tmp_path: Path, repackage_model_mocks: RepackageModelMocks
    ) -> None:
        TestRepackageModel.write_source(
            repackage_model_mocks, {"bar.safetensors": b"baz"}
        )
        with pytest.raises(ValueError, match="foo/foo_1_2_3.tar.gz has no model card"):
            LoRATrainer.repackage_model("foo", "foo_1_2_3", str(tmp_path))
        repackage_model_mocks._upload_archive.assert_not_called()


@dataclass
class ValidVersionMocks:
    list_s3_objects: MagicMock
    get_uploaded_model_folder_prefix: MagicMock


@pytest.fixture
def valid_version_mocks(mocker: MockerFixture) -> ValidVersionMocks:
    return ValidVersionMocks(
        mocker.patch("finetune.trainer.AWS.list_s3_objects"),
        mocker.patch.object(
            LoRATrainer,
            "_get_uploaded_model_folder_prefix",
            return_value="models/foo/foo_1_2_3",
        ),
    )


class TestValidModelCardVersion:
    def test_no_existing_objects_returns_true(
        self,
        valid_version_mocks: ValidVersionMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        valid_version_mocks.list_s3_objects.return_value = []
        assert make_base_trainer().valid_model_card_version(model_card)

    def test_existing_objects_returns_false(
        self,
        valid_version_mocks: ValidVersionMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        valid_version_mocks.list_s3_objects.return_value = [{"Key": "foo"}]
        assert not make_base_trainer().valid_model_card_version(model_card)

    def test_calls_list_s3_objects_with_correct_args(
        self,
        valid_version_mocks: ValidVersionMocks,
        model_card: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        valid_version_mocks.list_s3_objects.return_value = []
        make_base_trainer(
            aws_config={"bucket": "foo-bar", "region": "foo-bar-1", "role": "x"}
        ).valid_model_card_version(model_card)
        valid_version_mocks.list_s3_objects.assert_called_once_with(
            "foo-bar-1", "foo-bar", "models/foo/foo_1_2_3"
        )


class TestSerialise:
    @pytest.fixture(autouse=True)
    def install_schema_package_mock(self, mocker: MockerFixture) -> MagicMock:
        return mocker.patch("finetune.trainer.SchemaResolver.install_schema_package")

    def test_from_json_installs_schema_package_before_import(
        self,
        install_schema_package_mock: MagicMock,
        make_base_trainer: BaseTrainerFactory,
    ) -> None:
        LoRATrainer.from_json(make_base_trainer().to_json())
        install_schema_package_mock.assert_called_once_with(
            "londonaicentre-fixtures", "", True
        )

    def test_to_dict_stores_import_references(
        self, make_base_trainer: BaseTrainerFactory
    ) -> None:
        data = make_base_trainer().to_dict()
        assert data["schema"] == {"module": "fixtures", "qualname": "SchemaFixture"}
        assert data["prompt_builder"] == {
            "module": "conftest",
            "qualname": "PromptBuilderFixture",
        }

    def test_to_dict_embeds_config(self, make_base_trainer: BaseTrainerFactory) -> None:
        assert (
            make_base_trainer().to_dict()["config"]["training"]["base_model"] == "baz"
        )

    def test_to_json_is_single_line(
        self, make_base_trainer: BaseTrainerFactory
    ) -> None:
        assert "\n" not in make_base_trainer().to_json()

    def test_from_json_rebuilds_schema_by_import(
        self, make_base_trainer: BaseTrainerFactory
    ) -> None:
        assert (
            LoRATrainer.from_json(make_base_trainer().to_json()).schema is SchemaFixture
        )

    def test_from_json_rebuilds_prompt_builder_by_import(
        self, make_base_trainer: BaseTrainerFactory
    ) -> None:
        assert isinstance(
            LoRATrainer.from_json(make_base_trainer().to_json()).prompt_builder,
            PromptBuilderFixture,
        )

    def test_from_json_round_trips_fields(
        self, make_base_trainer: BaseTrainerFactory
    ) -> None:
        rebuilt = LoRATrainer.from_json(
            make_base_trainer(
                training_batch_names=["batch-a", "batch-b"],
                aws_config={"bucket": "x", "region": "y", "role": "z"},
                model_name="grault",
                description="garply",
            ).to_json()
        )
        assert rebuilt.training_batch_names == ["batch-a", "batch-b"]
        assert rebuilt.aws_config == {"bucket": "x", "region": "y", "role": "z"}
        assert rebuilt.model_name == "grault"
        assert rebuilt.description == "garply"
        assert rebuilt.base_model == "baz"

    def test_from_json_restores_job_id(
        self, make_base_trainer: BaseTrainerFactory
    ) -> None:
        trainer = make_base_trainer()
        assert LoRATrainer.from_json(trainer.to_json()).job_id == trainer.job_id

    def test_from_json_reads_from_file_path(
        self, tmp_path: Path, make_base_trainer: BaseTrainerFactory
    ) -> None:
        spec = tmp_path / "spec.json"
        spec.write_text(make_base_trainer(model_name="grault").to_json())
        assert LoRATrainer.from_json(str(spec)).model_name == "grault"

    def test_from_json_rebuilds_config_without_reading_file(
        self, mocker: MockerFixture, make_base_trainer: BaseTrainerFactory
    ) -> None:
        spec = make_base_trainer().to_json()
        load: MagicMock = mocker.patch("finetune.trainer.FinetuneConfig.load")
        assert LoRATrainer.from_json(spec).base_model == "baz"
        load.assert_not_called()


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
        make_base_trainer(
            model_name="waldo",
            aws_config={"bucket": "x", "region": "qux", "role": "y"},
        ).publish("target", model_card, True)
        archive.assert_called_once_with(
            "qux",
            Path("target"),
            "waldo",
            "foo_1_2_3",
            b"bar",
            "aicentre-nlpteam-mesa-public",
            False,
        )
