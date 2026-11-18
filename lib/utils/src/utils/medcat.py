from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from re import escape, finditer
from typing import Annotated, Any, NamedTuple, Protocol

from mesa_types import TrainingExample
from pydantic import BaseModel, PlainSerializer

type Timestamp = Annotated[
    datetime,
    PlainSerializer(lambda timestamp: timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")),
]


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
    acc: float
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
    irrelevant: bool
    create_time: Timestamp
    last_modified: Timestamp
    comment: str | None
    manually_created: bool
    acc: float
    meta_anns: dict[str, MetaAnnotation]


class MedCATTrainerExporter:
    """Export condition mentions as MedCATtrainer annotation records.

    Args:
        schema (Schema): The schema class whose output lists the mentions (e.g.
            conditionschema's `ConditionModel`).
        cui (str): The concept ID each annotation is given.
        user (str): The MedCATtrainer username each annotation is attributed to.

    """

    def __init__(self, schema: Schema, cui: str, user: str) -> None:
        self.__schema: Schema = schema
        self.__cui: str = cui
        self.__user: str = user

    def annotate(self, example: TrainingExample) -> list[Annotation] | None:
        """Map each mention in a training example to a MedCATtrainer annotation.

        Args:
            example (TrainingExample): The document and its schema output.

        Returns:
            list[Annotation] | None: An annotation per mention, in the order given,
                or None if the mentions cannot be located in the document.

        """
        located: list[tuple[Mention, Span]] | None = MedCATTrainerExporter.locate(
            example.content,
            list(self.__schema.model_validate(example.output).mentions),
        )
        if located is None:
            return None
        now: datetime = datetime.now()
        return [
            Annotation(
                user=self.__user,
                cui=self.__cui,
                value=mention.value,
                start=span.start,
                end=span.end,
                validated=True,
                correct=True,
                deleted=False,
                alternative=False,
                killed=False,
                irrelevant=False,
                create_time=now,
                last_modified=now,
                comment=None,
                manually_created=False,
                acc=1.0,
                meta_anns={
                    name: MetaAnnotation(
                        name=name, value=meta.value, acc=1.0, validated=True
                    )
                    for name, meta in (
                        ("Presence", mention.presence),
                        ("Subject", mention.subject),
                        ("Time", mention.time),
                    )
                },
            )
            for mention, span in located
        ]

    @staticmethod
    def locate(
        content: str, mentions: list[Mention]
    ) -> list[tuple[Mention, Span]] | None:
        """Locate each mention's value in a document as character offsets.

        Args:
            content (str): The document's text.
            mentions (list[Mention]): The document's mentions, in document order.

        Returns:
            list[tuple[Mention, Span]] | None: Each mention with its offsets, in the
                order given, or None if a value is not in the document as a whole
                word, or occurs more often than it is listed.

        """
        occurrences: dict[str, list[Span]] = {
            value: [
                Span(match.start(), match.start() + len(value))
                for match in finditer(rf"(?<!\w)(?={escape(value)}(?!\w))", content)
            ]
            for value in {mention.value for mention in mentions}
        }
        located: list[tuple[Mention, Span]] = []

        def unclaimed(span: Span) -> bool:
            return all(
                other.end <= span.start or span.end <= other.start
                for _, other in located
            )

        for mention in mentions:
            span: Span | None = next(
                (
                    occurrence
                    for occurrence in occurrences[mention.value]
                    if unclaimed(occurrence)
                ),
                None,
            )
            if span is None:
                return None
            located.append((mention, span))
        return (
            None
            if any(
                unclaimed(occurrence)
                for spans in occurrences.values()
                for occurrence in spans
            )
            else located
        )
