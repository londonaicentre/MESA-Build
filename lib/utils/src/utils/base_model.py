from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


class BaseModelFiles:
    """Fetches the files of a base model's checked revision."""

    @staticmethod
    def fetch(repo_id: str, revision: str) -> Path:
        """Download a base model's revision and verify it against the Hub's checksums.

        Args:
            repo_id (str): Hugging Face repo id of the base model.
            revision (str): Commit sha of the checked revision.

        Returns:
            Path: The local folder holding the verified files.

        Raises:
            ValueError: If a file is missing, unexpected or has a different checksum.

        """
        folder: Path = BaseModelFiles._download(repo_id, revision)
        mismatches: list[str] = BaseModelFiles._verify(repo_id, revision)
        if mismatches:
            raise ValueError(
                f"{repo_id} does not match revision {revision}: {', '.join(mismatches)}"
            )
        return folder

    @staticmethod
    def _download(repo_id: str, revision: str) -> Path:
        return Path(snapshot_download(repo_id, revision=revision))

    @staticmethod
    def _verify(repo_id: str, revision: str) -> list[str]:
        verification = HfApi().verify_repo_checksums(repo_id, revision=revision)
        return sorted(
            [mismatch["path"] for mismatch in verification.mismatches]
            + verification.missing_paths
            + verification.extra_paths
        )
