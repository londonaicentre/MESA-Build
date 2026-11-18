from collections.abc import Sequence
from enum import Enum
from typing import Any, NamedTuple, Protocol

from pydantic import BaseModel


class Mention(Protocol):
    value: str

    @property
    def presence(self) -> Enum: ...

    @property
    def subject(self) -> Enum: ...

    @property
    def time(self) -> Enum: ...


class Extraction(Protocol):
    @property
    def mentions(self) -> Sequence[Mention]: ...


class Schema(Protocol):
    def model_validate(self, obj: Any) -> Extraction: ...


class Span(NamedTuple):
    start: int
    end: int


class MetaAnnotation(BaseModel):
    name: str
    value: str
    validated: bool


class Annotation(BaseModel):
    user: str
    cui: str
    value: str
    start: int
    end: int
    validated: bool
    correct: bool
    deleted: bool
    alternative: bool
    killed: bool
    manually_created: bool
    acc: float
    meta_anns: dict[str, MetaAnnotation]


class Document(BaseModel):
    name: str
    text: str
    annotations: list[Annotation]


class Project(BaseModel):
    name: str
    cuis: str
    documents: list[Document]


class Export(BaseModel):
    projects: list[Project]
