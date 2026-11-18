import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mesa_types import TrainingExample
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from utils.medcat import (
    Annotation,
    MedCATTrainerExporter,
    Mention,
    MetaAnnotation,
    Span,
)


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


@dataclass
class ExportMocks:
    getLogger: MagicMock
    annotate: MagicMock


@pytest.fixture
def annotate_mocks(mocker: MockerFixture) -> AnnotateMocks:
    return AnnotateMocks(locate=mocker.patch.object(MedCATTrainerExporter, "locate"))


@pytest.fixture
def export_mocks(mocker: MockerFixture) -> ExportMocks:
    return ExportMocks(
        getLogger=mocker.patch("utils.medcat.getLogger"),
        annotate=mocker.patch.object(MedCATTrainerExporter, "annotate"),
    )


@pytest.fixture(scope="session")
def exporter() -> MedCATTrainerExporter:
    return MedCATTrainerExporter(FooExtraction, "123456", "foo", "bar")


@pytest.fixture
def logged_exporter(export_mocks: ExportMocks) -> MedCATTrainerExporter:
    return MedCATTrainerExporter(FooExtraction, "123456", "foo", "bar")


@pytest.fixture(scope="session")
def annotation() -> Annotation:
    return Annotation(
        user="foo",
        cui="123456",
        value="foo",
        start=0,
        end=3,
        validated=True,
        correct=True,
        deleted=False,
        alternative=False,
        killed=False,
        manually_created=False,
        acc=1.0,
        meta_anns={
            "Presence": MetaAnnotation(name="Presence", value="bar", validated=True)
        },
    )


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
        assert [annotation.model_dump() for annotation in annotations] == [
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
                "manually_created": False,
                "acc": 1.0,
                "meta_anns": {
                    name: {"name": name, "value": meta, "validated": True}
                    for name, meta in zip(("Presence", "Subject", "Time"), metas)
                },
            }
            for value, start, end, metas in [
                ("foo", 0, 3, ("bar", "baz", "bar")),
                ("qux", 9, 12, ("bar", "bar", "baz")),
            ]
        ]

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


class TestExport:
    def test_export_annotated_samples_writes_project(
        self,
        export_mocks: ExportMocks,
        logged_exporter: MedCATTrainerExporter,
        annotation: Annotation,
        tmp_path: Path,
    ) -> None:
        export_mocks.annotate.side_effect = [[], [annotation]]
        for name, content in (("foo", "foo then bar"), ("bar", "baz")):
            (tmp_path / f"{name}.json").write_text(
                TrainingExample(content=content, output={}).model_dump_json()
            )
        logged_exporter.export(tmp_path, tmp_path / "baz" / "qux.json")
        assert json.loads((tmp_path / "baz" / "qux.json").read_text()) == {
            "projects": [
                {
                    "name": "bar",
                    "cuis": "123456",
                    "documents": [
                        {"name": "bar", "text": "baz", "annotations": []},
                        {
                            "name": "foo",
                            "text": "foo then bar",
                            "annotations": [annotation.model_dump()],
                        },
                    ],
                }
            ]
        }
        export_mocks.getLogger.return_value.info.assert_called_once_with(
            "Exported %d documents and skipped %d", 2, 0
        )

    def test_export_unlocated_sample_skips_document_with_warning(
        self,
        export_mocks: ExportMocks,
        logged_exporter: MedCATTrainerExporter,
        annotation: Annotation,
        tmp_path: Path,
    ) -> None:
        export_mocks.annotate.side_effect = [None, [annotation]]
        for name in ("bar", "foo"):
            (tmp_path / f"{name}.json").write_text(
                TrainingExample(content="foo", output={}).model_dump_json()
            )
        logged_exporter.export(tmp_path, tmp_path / "baz.json")
        assert [
            document["name"]
            for document in json.loads((tmp_path / "baz.json").read_text())["projects"][
                0
            ]["documents"]
        ] == ["foo"]
        export_mocks.getLogger.return_value.warning.assert_called_once_with(
            "Skipped %s, as its mentions cannot be located", "bar"
        )
        export_mocks.getLogger.return_value.info.assert_called_once_with(
            "Exported %d documents and skipped %d", 1, 1
        )

    def test_export_invalid_sample_raises_validation_error(
        self, logged_exporter: MedCATTrainerExporter, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.json").write_text("{}")
        with pytest.raises(ValidationError):
            logged_exporter.export(tmp_path, tmp_path / "bar" / "baz.json")
