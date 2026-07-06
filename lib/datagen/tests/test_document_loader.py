from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from datagen.document_loader import DocumentLoader


@dataclass
class PathMocks:
    exists: MagicMock
    mkdir: MagicMock
    write_text: MagicMock


@dataclass
class DocumentLoaderDependencies:
    download_file: MagicMock
    tarfile_open: MagicMock
    model_validate_json: MagicMock


@pytest.fixture
def mock_path_operations(mocker: MockerFixture) -> PathMocks:
    return PathMocks(
        exists=mocker.patch.object(Path, "exists", return_value=False),
        mkdir=mocker.patch.object(Path, "mkdir"),
        write_text=mocker.patch.object(Path, "write_text"),
    )


@pytest.fixture
def mock_tar_member() -> MagicMock:
    member: MagicMock = MagicMock()
    member.name = "batch/document_001.json"
    return member


@pytest.fixture
def mock_dependencies(
    mocker: MockerFixture, mock_tar_member: MagicMock
) -> DocumentLoaderDependencies:
    mock_tar: MagicMock = MagicMock()
    mock_tar.__enter__ = MagicMock(return_value=mock_tar)
    mock_tar.__exit__ = MagicMock(return_value=False)
    mock_tar.getmembers.return_value = [mock_tar_member]
    mock_tar.extractfile.return_value = BytesIO(
        b'{"content": "foo", "source": "bar", "timestamp": "2026-01-01T00:00:00Z"}'
    )
    mock_document: MagicMock = MagicMock()
    mock_document.model_dump.return_value = {
        "content": "foo",
        "source": "bar",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    return DocumentLoaderDependencies(
        download_file=mocker.patch(
            "datagen.document_loader.AWS.download_file", return_value=True
        ),
        tarfile_open=mocker.patch(
            "datagen.document_loader.tarfile.open", return_value=mock_tar
        ),
        model_validate_json=mocker.patch(
            "datagen.document_loader.Document.model_validate_json",
            return_value=mock_document,
        ),
    )


def test_download_and_extract_cache_missing_downloads_file(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    DocumentLoader.download_and_extract("batch-2026-01-01-001.tar.gz", Path("output"))
    mock_dependencies.download_file.assert_called_once()


def test_download_and_extract_cache_exists_skips_download(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    mock_path_operations.exists.return_value = True
    DocumentLoader.download_and_extract("batch-2026-01-01-001.tar.gz", Path("output"))
    mock_dependencies.download_file.assert_not_called()


def test_download_and_extract_download_fails_raises_exception(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    mock_dependencies.download_file.return_value = False
    with pytest.raises(Exception, match="Failed to download"):
        DocumentLoader.download_and_extract(
            "batch-2026-01-01-001.tar.gz", Path("output")
        )


def test_download_and_extract_creates_output_directory(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    DocumentLoader.download_and_extract("batch-2026-01-01-001.tar.gz", Path("output"))
    mock_path_operations.mkdir.assert_any_call(parents=True, exist_ok=True)


@pytest.mark.parametrize(
    "filename,expected_mode",
    [("batch-2026-01-01-001.tar.gz", "r:gz"), ("batch-2026-01-01-001.tar", "r")],
)
def test_download_and_extract_file_extension_determines_tar_mode(
    mock_path_operations: PathMocks,
    mock_dependencies: DocumentLoaderDependencies,
    filename: str,
    expected_mode: str,
) -> None:
    mock_path_operations.exists.return_value = True
    DocumentLoader.download_and_extract(filename, Path("output"))
    mock_dependencies.tarfile_open.assert_called_once()
    assert mock_dependencies.tarfile_open.call_args[0][1] == expected_mode


def test_download_and_extract_valid_document_writes_to_output(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    DocumentLoader.download_and_extract("batch-2026-01-01-001.tar.gz", Path("output"))
    mock_path_operations.write_text.assert_called_once()


def test_download_and_extract_valid_document_returns_count(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    assert (
        DocumentLoader.download_and_extract(
            "batch-2026-01-01-001.tar.gz", Path("output")
        )
        == 1
    )


def test_download_and_extract_hashed_document_writes_to_output(
    mock_path_operations: PathMocks,
    mock_dependencies: DocumentLoaderDependencies,
    mock_tar_member: MagicMock,
) -> None:
    mock_tar_member.name = "batch/1a79a4d60de6718e8e5b326e338ae533.json"
    assert (
        DocumentLoader.download_and_extract(
            "batch-2026-01-01-001.tar.gz", Path("output")
        )
        == 1
    )
    mock_path_operations.write_text.assert_called_once()


def test_download_and_extract_non_document_member_skipped(
    mock_path_operations: PathMocks,
    mock_dependencies: DocumentLoaderDependencies,
    mock_tar_member: MagicMock,
) -> None:
    mock_tar_member.name = "batch-2026-01-01-001/readme.md"
    assert (
        DocumentLoader.download_and_extract(
            "batch-2026-01-01-001.tar.gz", Path("output")
        )
        == 0
    )
    mock_path_operations.write_text.assert_not_called()


@pytest.mark.parametrize(
    "member_name",
    [
        "batch-2026-01-01-001/other_001.json",
        "batch-2026-01-01-001/document_001.txt",
        "document_001",
        "metadata.json",
        "documentbatch.txt",
        "1a79a4d60de6718e8e5b326e338ae53.json",
        "1a79a4d60de6718e8e5b326e338ae533g.json",
    ],
)
def test_download_and_extract_invalid_member_pattern_skipped(
    mock_path_operations: PathMocks,
    mock_dependencies: DocumentLoaderDependencies,
    mock_tar_member: MagicMock,
    member_name: str,
) -> None:
    mock_tar_member.name = member_name
    assert (
        DocumentLoader.download_and_extract(
            "batch-2026-01-01-001.tar.gz", Path("output")
        )
        == 0
    )


def test_download_and_extract_extractfile_returns_none_skipped(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    mock_dependencies.tarfile_open.return_value.__enter__.return_value.extractfile.return_value = None
    assert (
        DocumentLoader.download_and_extract(
            "batch-2026-01-01-001.tar.gz", Path("output")
        )
        == 0
    )


def test_download_and_extract_multiple_documents_returns_correct_count(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    members: list[MagicMock] = [
        MagicMock(name=f"batch-2026-01-01-001/document_{i}.json") for i in range(3)
    ]
    for index, member in enumerate(members):
        member.name = f"batch-2026-01-01-001/document_{index}.json"
    mock_dependencies.tarfile_open.return_value.__enter__.return_value.getmembers.return_value = members
    assert (
        DocumentLoader.download_and_extract(
            "batch-2026-01-01-001.tar.gz", Path("output")
        )
        == 3
    )


def test_download_and_extract_s3_key_constructed_with_prefix(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    DocumentLoader.download_and_extract(
        "batch-2026-01-01-001.tar.gz", Path("output"), s3_prefix="custom/prefix"
    )
    assert (
        mock_dependencies.download_file.call_args[1]["object_name"]
        == "custom/prefix/batch-2026-01-01-001.tar.gz"
    )


def test_download_and_extract_empty_prefix_uses_filename_only(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    DocumentLoader.download_and_extract(
        "batch-2026-01-01-001.tar.gz", Path("output"), s3_prefix=""
    )
    assert (
        mock_dependencies.download_file.call_args[1]["object_name"]
        == "batch-2026-01-01-001.tar.gz"
    )


def test_download_and_extract_custom_bucket_used(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    DocumentLoader.download_and_extract(
        "batch-2026-01-01-001.tar.gz", Path("output"), bucket="foobar"
    )
    assert mock_dependencies.download_file.call_args[1]["bucket"] == "foobar"


def test_download_and_extract_custom_region_used(
    mock_path_operations: PathMocks, mock_dependencies: DocumentLoaderDependencies
) -> None:
    DocumentLoader.download_and_extract(
        "batch-2026-01-01-001.tar.gz", Path("output"), region="us-east-1"
    )
    assert mock_dependencies.download_file.call_args[1]["region_name"] == "us-east-1"


@pytest.fixture
def mock_list_s3_objects(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("datagen.document_loader.AWS.list_s3_objects")


def test_list_available_filenames_valid_keys_returns_filenames(
    mock_list_s3_objects: MagicMock,
) -> None:
    mock_list_s3_objects.return_value = [
        {"Key": "documents/batch-2026-01-01-001.tar.gz"},
        {"Key": "documents/batch-2026-01-02-001.tar"},
    ]
    assert DocumentLoader.list_available_document_batches() == [
        "batch-2026-01-01-001.tar.gz",
        "batch-2026-01-02-001.tar",
    ]


def test_list_available_filenames_non_archive_key_skipped(
    mock_list_s3_objects: MagicMock,
) -> None:
    mock_list_s3_objects.return_value = [
        {"Key": "documents/"},
        {"Key": "documents/readme.md"},
        {"Key": "documents/batch-2026-01-01-001.tar.gz"},
    ]
    assert DocumentLoader.list_available_document_batches() == [
        "batch-2026-01-01-001.tar.gz"
    ]


def test_list_available_filenames_empty_prefix_uses_key_directly(
    mock_list_s3_objects: MagicMock,
) -> None:
    mock_list_s3_objects.return_value = [{"Key": "batch-2026-01-01-001.tar.gz"}]
    assert DocumentLoader.list_available_document_batches(s3_prefix="") == [
        "batch-2026-01-01-001.tar.gz"
    ]


def test_list_available_filenames_passes_bucket_prefix_region(
    mock_list_s3_objects: MagicMock,
) -> None:
    mock_list_s3_objects.return_value = []
    DocumentLoader.list_available_document_batches(
        bucket="foobar", s3_prefix="custom", region="us-east-1"
    )
    mock_list_s3_objects.assert_called_once_with("us-east-1", "foobar", "custom/")
