import pytest
from pytest import MonkeyPatch
from unittest.mock import Mock
from pathlib import Path
from types import SimpleNamespace

from llamaserve.utils import Utils


def __setup_unzip_mocks(
    monkeypatch: MonkeyPatch, file_size: int, free_space: int
) -> None:
    mock_zip_instance: Mock = Mock()
    mock_zip_instance.testzip.return_value = None
    mock_zip_instance.infolist.return_value = [Mock(file_size=file_size)]
    monkeypatch.setattr(
        'zipfile.ZipFile',
        lambda source: Mock(
            __enter__=Mock(return_value=mock_zip_instance),
            __exit__=Mock(return_value=None),
        ),
    )
    monkeypatch.setattr(Path, 'exists', Mock(return_value=True))
    monkeypatch.setattr(
        'shutil.disk_usage',
        Mock(return_value=SimpleNamespace(free=free_space)),
    )


def test_unzip(monkeypatch: MonkeyPatch) -> None:
    __setup_unzip_mocks(monkeypatch, 100, 1000)
    assert Utils.unzip('foo.zip')


def test_unzip_insufficient_space(monkeypatch: MonkeyPatch) -> None:
    __setup_unzip_mocks(monkeypatch, 1000, 100)
    with pytest.raises(OSError):
        Utils.unzip('foo.zip')


def __setup_untar_mocks(monkeypatch: MonkeyPatch, file_size: int, free_space: int) -> None:
    mock_tar_instance = Mock()
    mock_tar_instance.getmembers.return_value = [
        Mock(size=file_size, isfile=Mock(return_value=True))
    ]
    monkeypatch.setattr(
        'tarfile.open',
        lambda source, mode='r:*': Mock(
            __enter__=Mock(return_value=mock_tar_instance),
            __exit__=Mock(return_value=None),
        ),
    )
    monkeypatch.setattr(Path, 'exists', Mock(return_value=True))
    monkeypatch.setattr(
        'shutil.disk_usage',
        Mock(return_value=SimpleNamespace(free=free_space)),
    )


def test_untar(monkeypatch: MonkeyPatch) -> None:
    __setup_untar_mocks(monkeypatch, 100, 1000)
    assert Utils.untar('foo.tar.gz')


def test_untar_insufficient_space(monkeypatch: MonkeyPatch) -> None:
    __setup_untar_mocks(monkeypatch, 1000, 100)
    with pytest.raises(OSError):
        Utils.untar('foo.tar.gz')


def test_one_file_true(monkeypatch: MonkeyPatch) -> None:
    mock_path = Mock(spec=Path)
    mock_path.is_file.return_value = True
    mock_path.parent.iterdir.return_value = [mock_path]
    monkeypatch.setattr('pathlib.Path', Mock(return_value=mock_path))
    assert Utils.one_file('bar.txt')


def test_one_file_false_multiple(monkeypatch: MonkeyPatch) -> None:
    mock_path = Mock(spec=Path)
    mock_other = Mock(spec=Path)
    mock_path.is_file.return_value = True
    mock_other.is_file.return_value = True
    mock_path.parent.iterdir.return_value = [mock_path, mock_other]
    monkeypatch.setattr('pathlib.Path', Mock(return_value=mock_path))
    assert not Utils.one_file('bar.txt')


def test_one_file_not_a_file(monkeypatch: MonkeyPatch) -> None:
    mock_path = Mock(spec=Path)
    mock_path.is_file.return_value = False
    monkeypatch.setattr('pathlib.Path', Mock(return_value=mock_path))
    with pytest.raises(ValueError):
        Utils.one_file('bar.txt')
