from pydantic import BaseModel


class BaseModelRecords(BaseModel):
    """Checked revision of each base model, keyed by Hugging Face repo id."""

    models: dict[str, str]


class BaseModelReport(BaseModel):
    """Outcome of checking a base model, recorded only if it has no findings."""

    repo_id: str
    revision: str
    findings: list[str]
