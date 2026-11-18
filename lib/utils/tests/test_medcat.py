from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock

import pytest
from mesa_types import TrainingExample
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from utils.medcat import Annotation, MedCATTrainerExporter, Mention, Span


class Foo(Enum):
    BAR = "bar"
    BAZ = "baz"


class FooMention(BaseModel):
    value: str
    presence: Foo = Foo.BAR
    subject: Foo = Foo.BAR
    time: Foo = Foo.BAR


class FooExtraction(BaseModel):
    mentions: list[FooMention]


@dataclass
class AnnotateMocks:
    locate: MagicMock


@pytest.fixture
def annotate_mocks(mocker: MockerFixture) -> AnnotateMocks:
    return AnnotateMocks(locate=mocker.patch.object(MedCATTrainerExporter, "locate"))


@pytest.fixture(scope="session")
def exporter() -> MedCATTrainerExporter:
    return MedCATTrainerExporter(FooExtraction, "123456", "foo")


class TestLocate:
    @pytest.mark.parametrize(
        ("content", "values", "spans"),
        [
            ("foo then bar", ["foo", "bar"], [Span(0, 3), Span(9, 12)]),
            ("foo, foo", ["foo", "foo"], [Span(0, 3), Span(5, 8)]),
            ("foo then bar", ["bar", "foo"], [Span(9, 12), Span(0, 3)]),
            ("baz foo then foo", ["baz foo", "foo"], [Span(0, 7), Span(13, 16)]),
            ("foobar foo", ["foo"], [Span(7, 10)]),
            ("foo: bar", ["foo"], [Span(0, 3)]),
            ("bar", [], []),
        ],
        ids=[
            "in_order",
            "repeats",
            "out_of_order",
            "overlapping_values",
            "skips_value_inside_word",
            "value_before_punctuation",
            "no_mentions",
        ],
    )
    def test_locate_every_occurrence_listed_returns_spans(
        self, content: str, values: list[str], spans: list[Span]
    ) -> None:
        mentions: list[Mention] = [FooMention(value=value) for value in values]
        assert MedCATTrainerExporter.locate(content, mentions) == list(
            zip(mentions, spans)
        )

    @pytest.mark.parametrize(
        ("content", "values"),
        [
            ("foo", ["bar"]),
            ("foo", ["foo", "foo"]),
            ("foobar", ["foo"]),
            ("Foo", ["foo"]),
            ("foo, foo", ["foo"]),
            ("foo bar foo", ["bar", "foo"]),
        ],
        ids=[
            "absent",
            "too_many_repeats",
            "only_inside_word",
            "case_differs",
            "occurrence_left_over",
            "occurrence_left_before_anchor",
        ],
    )
    def test_locate_mentions_not_matching_content_returns_none(
        self, content: str, values: list[str]
    ) -> None:
        mentions: list[Mention] = [FooMention(value=value) for value in values]
        assert MedCATTrainerExporter.locate(content, mentions) is None


class TestAnnotate:
    def test_annotate_located_mentions_returns_annotations(
        self, annotate_mocks: AnnotateMocks, exporter: MedCATTrainerExporter
    ) -> None:
        annotate_mocks.locate.return_value = [
            (FooMention(value="foo", subject=Foo.BAZ), Span(0, 3)),
            (FooMention(value="qux", time=Foo.BAZ), Span(9, 12)),
        ]
        annotations: list[Annotation] | None = exporter.annotate(
            TrainingExample(
                content="foo then qux",
                output={
                    "mentions": [
                        {"value": "foo", "presence": "bar", "subject": "baz"},
                        {"value": "qux", "time": "baz"},
                    ]
                },
            )
        )
        assert annotations is not None
        annotate_mocks.locate.assert_called_once_with(
            "foo then qux",
            [
                FooMention(value="foo", subject=Foo.BAZ),
                FooMention(value="qux", time=Foo.BAZ),
            ],
        )
        assert [
            annotation.model_dump(exclude={"create_time", "last_modified"})
            for annotation in annotations
        ] == [
            {
                "user": "foo",
                "cui": "123456",
                "value": value,
                "start": start,
                "end": end,
                "validated": True,
                "correct": True,
                "deleted": False,
                "alternative": False,
                "killed": False,
                "irrelevant": False,
                "comment": None,
                "manually_created": False,
                "acc": 1.0,
                "meta_anns": {
                    name: {"name": name, "value": meta, "acc": 1.0, "validated": True}
                    for name, meta in zip(("Presence", "Subject", "Time"), metas)
                },
            }
            for value, start, end, metas in [
                ("foo", 0, 3, ("bar", "baz", "bar")),
                ("qux", 9, 12, ("bar", "bar", "baz")),
            ]
        ]
        assert all(
            annotation.create_time == annotation.last_modified
            for annotation in annotations
        )

    def test_annotate_unlocated_mentions_returns_none(
        self, annotate_mocks: AnnotateMocks, exporter: MedCATTrainerExporter
    ) -> None:
        annotate_mocks.locate.return_value = None
        assert (
            exporter.annotate(
                TrainingExample(content="foo", output={"mentions": [{"value": "bar"}]})
            )
            is None
        )

    def test_annotate_invalid_output_raises_validation_error(
        self, exporter: MedCATTrainerExporter
    ) -> None:
        with pytest.raises(ValidationError):
            exporter.annotate(
                TrainingExample(content="foo", output={"mentions": [{"time": "bar"}]})
            )
