from dataclasses import dataclass

import pytest

from utils.medcat import MedCATTrainerExporter, Mention, Span


@dataclass
class FooMention:
    value: str


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
        mentions: list[Mention] = [FooMention(value) for value in values]
        assert MedCATTrainerExporter.locate(content, mentions) == spans

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
        mentions: list[Mention] = [FooMention(value) for value in values]
        assert MedCATTrainerExporter.locate(content, mentions) is None
