from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TestType(str, Enum):
    DNA = "DNA" # broad-spectrum sequencing approaches (e.g. WGS, WES, targeted panels, Sanger sequencing)
    FISH = "FISH" # Cytogenetic technique using fluorescent probes to detect specific chromosomal regions
    KARYOTYPE = "Karyotype" # Classical cytogenetic analysis of chromosome structure and number
    PCR = "PCR" # Amplification-based methods including qPCR, RT-PCR, digital PCR
    MLPA = "MLPA"# Copy number detection technique
    OTHER = "Other"


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
    MORBIDITY = "morbidity" # background diagnoses
    PATIENT_FINDING = "patient_finding" # symptoms, signs, or other observations
    FAMILY_HISTORY = "family_history" # morbidity or finding in a family member


class ClinicalFinding(BaseModel):
    type: ClinicalFindingType = Field(..., description="Type of clinical finding")
    value: str = Field(
        ...,
        description="Extracted descriptive text for the finding, remaining close to original text",
    )


class ClinicalContext(BaseModel):
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

class GeneIdentifier(BaseModel):
    nomenclature_system: str  # "HGNC", "Entrez", "Ensembl", etc.
    identifier: str           # The actual ID/symbol
    version: Optional[str]    # If there are version references

class QuantitativeResult(BaseModel):
    result_name: str = Field(
        ...,
        description="Type of measurement (e.g., 'Allele Frequency', 'Copy Number', etc)",
    )
    result_value: float = Field(..., description="The numeric value of the measurement")
    result_units: str = Field(..., description="Units of measurement where applicable")


class CategoricalResult(BaseModel):
    result_name: str = Field(
        ...,
        description="Classification system (e.g., 'Pathogenicity', 'Expression Level' etc)",
    )
    result_value: str = Field(
        ..., description="Assigned category (e.g., 'Pathogenic', 'Positive', 'High')"
    )


class BiomarkerTestResult(BaseModel):
    test_subject: str = Field(..., description="Person whose test is reported, e.g. patient, child, other relative")
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


class ClinicalOutcome(BaseModel):
    overall_implications: str = Field(
        ...,
        description="Interpretation of test significance for the patient",
    )
    overall_recommendations: str = Field(
        ...,
        description="Consolidated recommendations based on the complete panel of results",
    )


class GenomicTestReport(BaseModel):
    sufficient_data_quality: bool = Field(
        ...,
        description="This is True if the text is readable, and False if the text appears corrupted. It is not a reflection of content, but is here to flag poor text quality (e.g. OCR artefacts preventing text from being read)."
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
