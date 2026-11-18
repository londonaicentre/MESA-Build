import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mesa_types import TrainingExample
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from utils.medcat import MedCATTrainerExporter
from utils.medcat_types import (
    Annotation,
    Document,
    Export,
    Mention,
    MetaAnnotation,
    Project,
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
    read_examples: MagicMock
    annotate: MagicMock
    write_export: MagicMock


@pytest.fixture
def annotate_mocks(mocker: MockerFixture) -> AnnotateMocks:
    return AnnotateMocks(locate=mocker.patch.object(MedCATTrainerExporter, "locate"))


@pytest.fixture
def export_mocks(mocker: MockerFixture) -> ExportMocks:
    return ExportMocks(
        getLogger=mocker.patch("utils.medcat.getLogger"),
        read_examples=mocker.patch.object(MedCATTrainerExporter, "read_examples"),
        annotate=mocker.patch.object(MedCATTrainerExporter, "annotate"),
        write_export=mocker.patch.object(MedCATTrainerExporter, "write_export"),
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
    def test_export_annotated_examples_writes_project(
        self,
        export_mocks: ExportMocks,
        logged_exporter: MedCATTrainerExporter,
        annotation: Annotation,
    ) -> None:
        export_mocks.read_examples.return_value = {
            "bar": TrainingExample(content="baz", output={}),
            "foo": TrainingExample(content="foo then bar", output={}),
        }
        export_mocks.annotate.side_effect = [[], [annotation]]
        logged_exporter.export(Path("qux"), Path("quux.json"))
        export_mocks.read_examples.assert_called_once_with(Path("qux"))
        export_mocks.write_export.assert_called_once_with(
            Export(
                projects=[
                    Project(
                        name="bar",
                        cuis="123456",
                        documents=[
                            Document(name="bar", text="baz", annotations=[]),
                            Document(
                                name="foo",
                                text="foo then bar",
                                annotations=[annotation],
                            ),
                        ],
                    )
                ]
            ),
            Path("quux.json"),
        )
        export_mocks.getLogger.return_value.info.assert_called_once_with(
            "Exported %d documents and skipped %d", 2, 0
        )

    def test_export_unlocated_example_skips_document_with_warning(
        self,
        export_mocks: ExportMocks,
        logged_exporter: MedCATTrainerExporter,
        annotation: Annotation,
    ) -> None:
        export_mocks.read_examples.return_value = {
            "bar": TrainingExample(content="foo", output={}),
            "foo": TrainingExample(content="foo", output={}),
        }
        export_mocks.annotate.side_effect = [None, [annotation]]
        logged_exporter.export(Path("qux"), Path("quux.json"))
        export_mocks.write_export.assert_called_once_with(
            Export(
                projects=[
                    Project(
                        name="bar",
                        cuis="123456",
                        documents=[
                            Document(name="foo", text="foo", annotations=[annotation])
                        ],
                    )
                ]
            ),
            Path("quux.json"),
        )
        export_mocks.getLogger.return_value.warning.assert_called_once_with(
            "Skipped %s, as its mentions cannot be located", "bar"
        )
        export_mocks.getLogger.return_value.info.assert_called_once_with(
            "Exported %d documents and skipped %d", 1, 1
        )


@pytest.mark.integration
class TestReadExamples:
    def test_read_examples_folder_returns_examples_by_name(
        self, tmp_path: Path
    ) -> None:
        for name in ("foo", "bar"):
            (tmp_path / f"{name}.json").write_text(
                TrainingExample(content=name, output={}).model_dump_json()
            )
        (tmp_path / "baz.txt").write_text("qux")
        assert MedCATTrainerExporter.read_examples(tmp_path) == {
            "bar": TrainingExample(content="bar", output={}),
            "foo": TrainingExample(content="foo", output={}),
        }
        assert list(MedCATTrainerExporter.read_examples(tmp_path)) == ["bar", "foo"]

    def test_read_examples_invalid_file_raises_validation_error(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.json").write_text("{}")
        with pytest.raises(ValidationError):
            MedCATTrainerExporter.read_examples(tmp_path)


@pytest.mark.integration
class TestWriteExport:
    def test_write_export_new_folder_writes_json(
        self, annotation: Annotation, tmp_path: Path
    ) -> None:
        export: Export = Export(
            projects=[
                Project(
                    name="foo",
                    cuis="123456",
                    documents=[
                        Document(name="bar", text="foo", annotations=[annotation])
                    ],
                )
            ]
        )
        MedCATTrainerExporter.write_export(export, tmp_path / "baz" / "qux.json")
        assert (
            json.loads((tmp_path / "baz" / "qux.json").read_text())
            == export.model_dump()
        )
