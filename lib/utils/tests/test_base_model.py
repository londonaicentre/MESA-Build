from dataclasses import dataclass
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import pytest
from huggingface_hub import constants
from pytest_mock import MockerFixture

from utils.base_model import BaseModelFiles


class BaseModelFilesFixture(BaseModelFiles):
    @staticmethod
    def download(repo_id: str, revision: str) -> Path:
        return BaseModelFiles._download(repo_id, revision)

    @staticmethod
    def verify(repo_id: str, revision: str) -> list[str]:
        return BaseModelFiles._verify(repo_id, revision)


@dataclass
class FetchMocks:
    _download: MagicMock
    _verify: MagicMock


@pytest.fixture
def fetch_mocks(mocker: MockerFixture) -> FetchMocks:
    return FetchMocks(
        _download=mocker.patch.object(
            BaseModelFiles, "_download", return_value=Path("/foo")
        ),
        _verify=mocker.patch.object(BaseModelFiles, "_verify", return_value=[]),
    )


class TestFetch:
    def test_fetch_verified_download_returns_folder(
        self, fetch_mocks: FetchMocks
    ) -> None:
        assert BaseModelFiles.fetch("foo/bar", "baz") == Path("/foo")
        fetch_mocks._download.assert_called_once_with("foo/bar", "baz")
        fetch_mocks._verify.assert_called_once_with("foo/bar", "baz")

    def test_fetch_mismatched_download_raises_naming_files(
        self, fetch_mocks: FetchMocks
    ) -> None:
        fetch_mocks._verify.return_value = ["qux", "quux"]

        with pytest.raises(ValueError, match="revision baz: qux, quux$"):
            BaseModelFiles.fetch("foo/bar", "baz")


@pytest.mark.integration
class TestDownload:
    def test_download_cached_revision_returns_snapshot_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        revision: str = "0123456789abcdef0123456789abcdef01234567"
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
        snapshot: Path = tmp_path / "models--foo--bar" / "snapshots" / revision
        snapshot.mkdir(parents=True)

        assert BaseModelFilesFixture.download("foo/bar", revision) == snapshot


@pytest.mark.integration
class TestVerify:
    __REPO_ID: Final = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    __REVISION: Final = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"

    @pytest.fixture
    def snapshot(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
        return BaseModelFilesFixture.download(
            TestVerify.__REPO_ID, TestVerify.__REVISION
        )

    def test_verify_untouched_snapshot_returns_empty(self, snapshot: Path) -> None:
        assert (
            BaseModelFilesFixture.verify(TestVerify.__REPO_ID, TestVerify.__REVISION)
            == []
        )

    def test_verify_changed_and_extra_files_returns_their_paths(
        self, snapshot: Path
    ) -> None:
        (snapshot / "config.json").unlink()
        (snapshot / "config.json").write_text("{}")
        (snapshot / "foo.py").write_text("")

        assert BaseModelFilesFixture.verify(
            TestVerify.__REPO_ID, TestVerify.__REVISION
        ) == [
            "config.json",
            "foo.py",
        ]
