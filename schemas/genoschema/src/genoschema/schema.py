from enum import Enum
from typing import List, Optional, Self, Any

from pydantic import BaseModel, Field, model_validator


class CaseInsensitiveEnumModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def lowercase_enums(cls: type[Self], data: dict[str, Any]) -> dict[str, Any]:
        for name, info in cls.model_fields.items():
            annotation, value = info.annotation, data.get(name)
            if (
                isinstance(annotation, type)
                and issubclass(annotation, Enum)
                and isinstance(value, str)
            ):
                data[name] = value.lower()
        return data


class TestType(str, Enum):
    DNA = "dna"  # broad-spectrum sequencing approaches (e.g. WGS, WES, targeted panels, Sanger sequencing)
    FISH = "fish"  # Cytogenetic technique using fluorescent probes to detect specific chromosomal regions
    KARYOTYPE = (
        "karyotype"  # Classical cytogenetic analysis of chromosome structure and number
    )
    PCR = "pcr"  # Amplification-based methods including qPCR, RT-PCR, digital PCR
    MLPA = "mlpa"  # Copy number detection technique
    OTHER = "other"


class ResultStatus(str, Enum):
    ABNORMAL = "abnormal"
    NORMAL = "normal"
    UNCERTAIN_SIGNIFICANCE = "uncertain_significance"
    FAILED_OR_INCONCLUSIVE = "failed_or_inconclusive"


class ResultEntityType(str, Enum):
    CHROMOSOME = "chromosome"
    GENE = "gene"
    EXON = "exon"
    VARIANT = "variant"
    PROTEIN = "protein"


class ClinicalFindingType(str, Enum):
    MORBIDITY = "morbidity"  # background diagnoses
    PATIENT_FINDING = "patient_finding"  # symptoms, signs, or other observations
    FAMILY_HISTORY = "family_history"  # morbidity or finding in a family member


class ClinicalFinding(CaseInsensitiveEnumModel):
    type: ClinicalFindingType = Field(..., description="Type of clinical finding")
    value: str = Field(
        ...,
        description="Extracted descriptive text for the finding, remaining close to original text",
    )


class ClinicalContext(CaseInsensitiveEnumModel):
    referral_reason: str = Field(
        ...,
        description="The reason given for genomic test referral, including suspected diagnosis or clinical question",
    )
    test_clinical_rationale: Optional[str] = Field(
        ...,
        description="The anticipated clinical insights as rationale for performing this test",
    )
    clinical_findings: List[ClinicalFinding] = Field(
        default_factory=list, description="Relevant historical patient information"
    )


class GeneIdentifier(CaseInsensitiveEnumModel):
    nomenclature_system: str = Field(
        description="Gene nomenclature system (e.g., HGNC, Entrez, Ensembl)"
    )
    identifier: str = Field(description="The actual gene ID or symbol")
    version: Optional[str] = Field(
        default=None, description="Version reference for the gene identifier"
    )


class QuantitativeResult(CaseInsensitiveEnumModel):
    result_name: str = Field(
        ...,
        description="Type of measurement (e.g., 'Allele Frequency', 'Copy Number', etc)",
    )
    result_value: float = Field(..., description="The numeric value of the measurement")
    result_units: str = Field(..., description="Units of measurement where applicable")


class CategoricalResult(CaseInsensitiveEnumModel):
    result_name: str = Field(
        ...,
        description="Classification system (e.g., 'Pathogenicity', 'Expression Level' etc)",
    )
    result_value: str = Field(
        ..., description="Assigned category (e.g., 'Pathogenic', 'Positive', 'High')"
    )


class BiomarkerTestResult(CaseInsensitiveEnumModel):
    test_subject: str = Field(
        ...,
        description="Person whose test is reported, e.g. patient, child, other relative",
    )
    test_type: TestType = Field(
        ..., description="The category of genomic test performed"
    )
    other_test_type: Optional[str] = Field(
        None, description="Description of test type if 'Other' is selected"
    )
    test_methodology: str = Field(
        ..., description="Technical description of the test methodology"
    )
    sample_origin: Optional[str] = Field(
        None, description="Anatomical source of the sample tested"
    )
    result_entity_type: ResultEntityType = Field(
        ..., description="The type of the primary entity being reported"
    )
    result_entity: str = Field(
        ..., description="The name of the primary entity being reported"
    )
    gene_nomenclature: List[GeneIdentifier] = Field(
        default_factory=list,
        description="Gene nomenclatures used",
    )
    result_region: Optional[str] = Field(
        None, description="The specific region or variant within entity being reported"
    )
    result_status: ResultStatus = Field(
        ..., description="Overall status of the test result"
    )
    result_description: str = Field(
        ..., description="Textual description of the findings as reported"
    )
    quantitative_results: List[QuantitativeResult] = Field(
        default_factory=list,
        description="Numeric measurements associated with the result",
    )
    categorical_results: List[CategoricalResult] = Field(
        default_factory=list, description="Categorical classifications of the finding"
    )
    clinical_implications: Optional[str] = Field(
        None,
        description="Disease associations or risk factors specific to this test result",
    )


class ClinicalOutcome(CaseInsensitiveEnumModel):
    overall_implications: str = Field(
        ...,
        description="Interpretation of test significance for the patient",
    )
    overall_recommendations: str = Field(
        ...,
        description="Consolidated recommendations based on the complete panel of results",
    )


class GenomicTestReport(CaseInsensitiveEnumModel):
    sufficient_data_quality: bool = Field(
        ...,
        description="This is True if the text is readable, and False if the text appears corrupted. It is not a reflection of content, but is here to flag poor text quality (e.g. OCR artefacts preventing text from being read).",
    )
    is_genomic_report: bool = Field(
        ...,
        description="Is the document a genomic test report?",
    )
    clinical_context: Optional[ClinicalContext] = Field(
        ..., description="Clinical context for the test"
    )
    biomarker_test_results: Optional[List[BiomarkerTestResult]] = Field(
        ..., description="Results of individual biomarker tests performed"
    )
    clinical_outcome: Optional[ClinicalOutcome] = Field(
        ..., description="Overall interpretation and recommendations"
    )
