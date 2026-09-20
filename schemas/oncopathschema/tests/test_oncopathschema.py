"""Tests for oncopathschema package."""

import pytest
from pydantic import ValidationError

from oncopathschema.prompt_builder import PromptBuilder
from oncopathschema.schema import (
    BiomarkerStatus,
    FeatureFinding,
    FeatureStatus,
    FindingFeature,
    FindingStatus,
    InvasionStatus,
    OncoPathModel,
    Specimen,
    SpecimenFinding,
    TreatmentResponseStatus,
)


def test_validate_schema() -> None:
    """Test that we can instantiate and validate the schema."""
    OncoPathModel.model_json_schema()


def _finding(
    status: FindingStatus,
    features: list[FeatureFinding] | None = None,
    invasion_status: InvasionStatus | None = None,
) -> SpecimenFinding:
    return SpecimenFinding(
        is_lymph_node=False,
        finding_status=status,
        finding_summary="summary",
        features=features or [],
        morphology=None,
        invasion_status=invasion_status,
        tumour_nature=None,
        differentiation=None,
        dimensions_desc=None,
    )


def test_finding_exists_without_malignancy() -> None:
    """A finding can be created for benign or normal tissue.

    The restructure that domain rule 2 previously forbade: a benign or normal
    specimen must be able to carry a finding, and so a score or feature.
    """
    finding = _finding(FindingStatus.NOT_CANCEROUS)
    assert finding.morphology is None
    assert finding.scores == []


def test_finding_status_has_no_borderline_or_normal_tissue_value() -> None:
    """BORDERLINE_MALIGNANT_POTENTIAL and NORMAL_TISSUE were reverted.

    Report text rarely signals "certain diagnosis, uncertain behaviour" vs. an
    unresolved diagnosis, or "normal" vs. "a named benign finding", clearly enough
    for consistent labelling - both fold back into UNCERTAIN / NOT_CANCEROUS.
    """
    assert not hasattr(FindingStatus, "BORDERLINE_MALIGNANT_POTENTIAL")
    assert not hasattr(FindingStatus, "NORMAL_TISSUE")
    assert len(FindingStatus) == 3


def test_feature_records_absence() -> None:
    """A feature stated absent is distinguishable from a feature never mentioned."""
    stated_absent = _finding(
        FindingStatus.CANCEROUS,
        features=[
            FeatureFinding(
                feature=FindingFeature.LYMPHOVASCULAR_INVASION,
                feature_status=FeatureStatus.ABSENT,
            )
        ],
    )
    never_mentioned = _finding(FindingStatus.CANCEROUS)

    assert stated_absent.features[0].feature_status is FeatureStatus.ABSENT
    assert never_mentioned.features == []
    assert stated_absent.features != never_mentioned.features


def test_invasive_and_in_situ_combination_removed() -> None:
    """An invasive + in-situ report is two findings, so no combined enum value."""
    assert not hasattr(InvasionStatus, "INVASIVE_AND_IN_SITU")
    with pytest.raises(ValidationError):
        _finding(
            FindingStatus.CANCEROUS,
            invasion_status="invasive_and_in_situ",  # type: ignore[arg-type]
        )


def test_specimen_defaults() -> None:
    """New specimen fields default to the 'nothing stated' reading."""
    specimen = Specimen.model_validate(
        {
            "anatomical_site": "breast",
            "anatomical_site_name_desc": "Left breast, wide local excision",
        }
    )
    assert specimen.is_sentinel is False
    assert specimen.is_multifocal is False
    assert specimen.is_multicentric is False
    assert specimen.treatment_response is TreatmentResponseStatus.NOT_STATED
    assert specimen.findings == []


def test_benign_feature_vocabulary_merged() -> None:
    """Benign findings share one feature vocabulary with cancer findings."""
    assert FindingFeature.DYSPLASIA in FindingFeature
    assert FindingFeature.LYMPHOVASCULAR_INVASION in FindingFeature
    assert FindingFeature.COMEDO_NECROSIS in FindingFeature
    # NORMAL_UNREMARKABLE is now FindingStatus.NOT_CANCEROUS's job
    assert not hasattr(FindingFeature, "NORMAL_UNREMARKABLE")


def test_multifocality_is_specimen_level_not_a_feature() -> None:
    """Multifocal/multicentric moved from FindingFeature to Specimen booleans.

    This is a specimen-wide signal recorded alongside, not instead of, per-focus
    findings - a report that itemises each focus (its own size/site/grade) still
    gets one SpecimenFinding per focus.
    """
    assert not hasattr(FindingFeature, "MULTIFOCAL")
    assert not hasattr(FindingFeature, "MULTICENTRIC")
    specimen = Specimen.model_validate(
        {
            "anatomical_site": "breast",
            "anatomical_site_name_desc": "Left breast, wide local excision",
            "is_multifocal": True,
        }
    )
    assert specimen.is_multifocal is True
    assert specimen.is_multicentric is False


def test_biomarker_status_has_no_hypothetical_value() -> None:
    """Only confirmed biomarker results are reportable; pending tests are omitted
    entirely rather than recorded with a placeholder status."""
    assert not hasattr(BiomarkerStatus, "HYPOTHETICAL")


def test_build_datagen_prompt() -> None:
    """Test building data generation prompt."""
    builder = PromptBuilder()
    prompt = builder.build_datagen_prompt()

    # Placeholders replaced
    assert "{SCHEMA}" not in prompt, "Schema placeholder should be replaced"
    assert "{EXAMPLE}" not in prompt, "Example placeholder should be replaced"

    # Schema source present
    assert "OncoPathModel" in prompt, "Schema should contain the root model"
    assert "is_malignancy_identified_on_specimen" in prompt, (
        "Schema should contain key field"
    )

    # Example should be present
    assert '"content"' in prompt or "'content'" in prompt, (
        "Example should contain 'content' field"
    )
    assert '"output"' in prompt or "'output'" in prompt, (
        "Example should contain 'output' field"
    )


def test_build_main_prompt() -> None:
    """Test building main prompt."""
    builder = PromptBuilder()
    prompt = builder.build_main_prompt()

    # Placeholders replaced
    assert "{SCHEMA}" not in prompt, "Schema placeholder should be replaced"
    assert "{EXAMPLE}" not in prompt, "Main prompt should not have example placeholder"

    # Schema source present
    assert "OncoPathModel" in prompt, "Schema should contain the root model"
    assert "is_malignancy_identified_on_specimen" in prompt, (
        "Schema should contain key field"
    )
