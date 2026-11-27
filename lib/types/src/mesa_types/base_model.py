from pydantic import BaseModel


class BaseModelRecord(BaseModel):
    """Checked revision of a base model, with the sha256 of each of its files."""

    revision: str
    files: dict[str, str]


class BaseModelRecords(BaseModel):
    """Checked base models, keyed by Hugging Face repo id."""

    models: dict[str, BaseModelRecord]


class BaseModelReport(BaseModel):
    """Outcome of checking a base model, recorded only if it has no findings."""

    repo_id: str
    record: BaseModelRecord
    findings: list[str]
