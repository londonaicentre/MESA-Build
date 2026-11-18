from re import escape, finditer
from typing import NamedTuple, Protocol


class Mention(Protocol):
    value: str


class Span(NamedTuple):
    start: int
    end: int


class MedCATTrainerExporter:
    """Export condition mentions as MedCATtrainer annotation records."""

    @staticmethod
    def locate(content: str, mentions: list[Mention]) -> list[Span] | None:
        """Locate each mention's value in a document as character offsets.

        Args:
            content (str): The document's text.
            mentions (list[Mention]): The document's mentions, in document order.

        Returns:
            list[Span] | None: Each mention's offsets, in the order given, or None if
                a value is not in the document as a whole word, or occurs more often
                than it is listed.

        """
        occurrences: dict[str, list[Span]] = {
            value: [
                Span(match.start(), match.start() + len(value))
                for match in finditer(rf"(?<!\w)(?={escape(value)}(?!\w))", content)
            ]
            for value in {mention.value for mention in mentions}
        }
        located: list[Span] = []

        def unclaimed(span: Span) -> bool:
            return all(
                other.end <= span.start or span.end <= other.start for other in located
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
            located.append(span)
        return (
            None
            if any(
                unclaimed(occurrence)
                for spans in occurrences.values()
                for occurrence in spans
            )
            else located
        )
