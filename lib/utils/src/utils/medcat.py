from logging import Logger, getLogger
from pathlib import Path
from re import escape, finditer

from mesa_types import TrainingExample

from utils.medcat_types import (
    Annotation,
    Document,
    Export,
    Mention,
    MetaAnnotation,
    Project,
    Schema,
    Span,
)


class MedCATTrainerExporter:
    """Export condition mentions as a MedCATtrainer project, holding only the fields its importer reads rather than a full MedCATtrainer export.

    Args:
        schema (Schema): The schema class whose output lists the mentions (e.g.
            conditionschema's `ConditionModel`).
        cui (str): The concept ID each annotation is given.
        user (str): The MedCATtrainer username each annotation is attributed to.
        project_name (str): The name of the exported project.

    """

    def __init__(self, schema: Schema, cui: str, user: str, project_name: str) -> None:
        self.__schema: Schema = schema
        self.__cui: str = cui
        self.__user: str = user
        self.__project_name: str = project_name
        self.__logger: Logger = getLogger(__name__)

    def export(self, input_folder: Path, output_file: Path) -> None:
        """Export a folder of training examples as a MedCATtrainer project.

        Args:
            input_folder (Path): The folder of training example JSON files.
            output_file (Path): The file the project is written to.

        """
        examples: dict[str, TrainingExample] = {
            path.stem: TrainingExample.model_validate_json(path.read_text())
            for path in sorted(input_folder.glob("*.json"))
        }
        annotations: dict[str, list[Annotation] | None] = {
            name: self.annotate(example) for name, example in examples.items()
        }
        for name in [name for name, located in annotations.items() if located is None]:
            self.__logger.warning("Skipped %s, as its mentions cannot be located", name)
        documents: list[Document] = [
            Document(name=name, text=examples[name].content, annotations=located)
            for name, located in annotations.items()
            if located is not None
        ]
        self.__logger.info(
            "Exported %d documents and skipped %d",
            len(documents),
            len(examples) - len(documents),
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            Export(
                projects=[
                    Project(
                        name=self.__project_name, cuis=self.__cui, documents=documents
                    )
                ]
            ).model_dump_json()
        )

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
                manually_created=False,
                acc=1.0,
                meta_anns={
                    name: MetaAnnotation(name=name, value=meta.value, validated=True)
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
