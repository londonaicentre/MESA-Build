from abc import ABC, abstractmethod
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import cast

from litellm import ConfigDict
from pydantic import BaseModel
import yaml


class Profile(BaseModel):
    profile_id: str | None = None
    morphology: str = ""
    descriptive_name: str = ""
    biomarker_profile: str = ""

    model_config = ConfigDict(
        extra="allow",
    )


class Profiles(BaseModel):
    items: dict[str, Profile] = {}

    def __init__(self, file_path: Traversable) -> None:
        super().__init__(items=yaml.safe_load(file_path.read_text()))


class SamplingAttribute(BaseModel):
    probability: float
    description: str


class SamplingElement(BaseModel):
    mutually_exclusive: bool


class DataQualityIssues(SamplingElement):
    none: SamplingAttribute
    missing_dates: SamplingAttribute
    missing_stage_information: SamplingAttribute
    misspelled_terms: SamplingAttribute
    inconsistent_formatting: SamplingAttribute
    truncated_content: SamplingAttribute
    corrupted_values: SamplingAttribute
    missing_sections: SamplingAttribute


class RedactionStyle(SamplingElement):
    fully_redacted: SamplingAttribute
    partially_redacted: SamplingAttribute
    not_redacted: SamplingAttribute


class FooterStyle(SamplingElement):
    complex_footer: SamplingAttribute
    formal_footer: SamplingAttribute
    basic_footer: SamplingAttribute
    no_footer: SamplingAttribute


class HeaderStyle(SamplingElement):
    complex_header: SamplingAttribute
    standard_header: SamplingAttribute
    minimal_header: SamplingAttribute
    no_header: SamplingAttribute


class WritingStyle(SamplingElement):
    concise_factual: SamplingAttribute
    verbose_professional: SamplingAttribute
    narrative_empathetic: SamplingAttribute


class ContentType(SamplingElement):
    initial_consultation: SamplingAttribute
    follow_up_stable: SamplingAttribute
    treatment_review: SamplingAttribute
    treatment_change: SamplingAttribute
    disease_change: SamplingAttribute
    mdt_outcome: SamplingAttribute
    post_acute: SamplingAttribute


class DocumentType(SamplingElement):
    is_a_letter: SamplingAttribute
    is_not_a_letter: SamplingAttribute


class Style(BaseModel):
    document_type: DocumentType
    content_type: ContentType
    writing_style: WritingStyle
    header_style: HeaderStyle
    footer_style: FooterStyle
    redaction_style: RedactionStyle
    data_quality_issues: DataQualityIssues

    def __init__(self, file_path: Traversable) -> None:
        super().__init__(**yaml.safe_load(file_path.read_text()))


class TimelineEvents(SamplingElement):
    clinical_trial_consideration: SamplingAttribute
    clinical_trial_enrollment: SamplingAttribute
    patient_death: SamplingAttribute
    __pydantic_extra__: dict[str, SamplingAttribute]

    model_config = ConfigDict(
        extra="allow",
    )


class PatientFindings(SamplingElement):
    comorbidity: SamplingAttribute
    social_history: SamplingAttribute
    family_history: SamplingAttribute
    symptoms_present: SamplingAttribute
    symptoms_resolved: SamplingAttribute
    physical_exam_positive: SamplingAttribute


class FunctionalStatus(SamplingElement):
    good: SamplingAttribute
    poor: SamplingAttribute
    omit: SamplingAttribute


class MentalState(SamplingElement):
    positive: SamplingAttribute
    distressed: SamplingAttribute
    omit: SamplingAttribute


class DiagnosisDate(SamplingElement):
    include_year_and_month: SamplingAttribute
    include_year_only: SamplingAttribute
    omit: SamplingAttribute


class DiseaseTrajectory(SamplingElement):
    stable: SamplingAttribute
    improving: SamplingAttribute
    slowly_progressing: SamplingAttribute
    rapidly_deteriorating: SamplingAttribute
    remission: SamplingAttribute


class TreatmentComplexity(SamplingElement):
    simple: SamplingAttribute
    moderate: SamplingAttribute
    complex: SamplingAttribute


class Content(BaseModel):
    timeline_events: TimelineEvents
    patient_findings: PatientFindings
    functional_status: FunctionalStatus
    mental_state: MentalState
    diagnosis_date: DiagnosisDate
    disease_trajectory: DiseaseTrajectory
    treatment_complexity: TreatmentComplexity
    __pydantic_extra__: dict[str, dict[str, bool | SamplingAttribute]]

    model_config = ConfigDict(
        extra="allow",
    )

    def __init__(self, file_path: Traversable) -> None:
        super().__init__(**yaml.safe_load(file_path.read_text()))


class SchemaLlamaAssets(ABC):
    def __init__(self, base_dir: str) -> None:
        self.all_profiles: list[Profile] = []
        self.base_dir: Traversable = files(base_dir)

    def load_prompt_template(self, template_name: str = "default") -> str:
        return self.base_dir.joinpath("prompts/" + f"{template_name}.md").read_text()

    def load_all_profiles(self) -> list[Profile]:
        items: list[Traversable] = cast(
            list[Traversable], sorted(self.base_dir.joinpath("profiles").iterdir())
        )
        item: Traversable
        for item in items:
            if item.is_file() and item.name.endswith(".yml"):
                profiles: list[Profile] = self._load_profiles_from_file(item)
                self.all_profiles.extend(profiles)
        return self.all_profiles

    @abstractmethod
    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        pass

    def load_profiles_from_files(self, filenames: list[str]) -> list[Profile]:
        for filename in filenames:
            file_path: str = "profiles/" + filename
            try:
                profiles: list[Profile] = self._load_profiles_from_file(
                    self.base_dir.joinpath(file_path)
                )
                self.all_profiles.extend(profiles)
            except FileNotFoundError:
                raise FileNotFoundError(f"Profile file not found: {file_path}")
        return self.all_profiles

    def get_profile_count(self) -> int:
        return len(self.all_profiles)

    @abstractmethod
    def format_profile_prompt(self, profile: Profile) -> str:
        pass

    def load_style_data(self) -> Style:
        return Style(self.base_dir.joinpath("style.yml"))

    def load_content_data(self) -> Content:
        return Content(self.base_dir.joinpath("content.yml"))

    def load_structures(self, enabled_structures: list[str]) -> dict[str, str]:
        self.structures: dict[str, str] = {}
        for filename in enabled_structures:
            file_path: str = "structure/" + filename
            try:
                self.structures[filename] = self.base_dir.joinpath(
                    file_path
                ).read_text()
            except FileNotFoundError:
                raise FileNotFoundError(f"Structure file not found: {file_path}")
        return self.structures

    def get_structure_name_without_extension(self, filename: str) -> str:
        return Path(self.base_dir.joinpath("structure/" + filename).name).stem 
