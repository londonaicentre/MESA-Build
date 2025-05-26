from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TestType(str, Enum):
    DNA = "dna"
    FISH = "fish"
    KARYOTYPE = "karyotype"
    PCR = "pcr"
    MLPA = "mlpa"
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
    GENOMIC_FEATURE = "genomic_feature"
    PANEL = "panel"


class ClinicalFindingType(str, Enum):
    MORBIDITY = "morbidity"
    PATIENT_FINDING = "patient_finding"
    FAMILY_HISTORY = "family_history"


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
    test_clinical_rationale: str = Field(
        ...,
        description="The anticipated clinical insights as rationale for performing this test",
    )
    clinical_findings: List[ClinicalFinding] = Field(
        default_factory=list, description="Relevant historical patient information"
    )


class QuantitativeResult(BaseModel):
    result_name: str = Field(
        ...,
        description="Type of measurement (e.g., 'Allele Frequency', 'Copy Number', etc)",
    )
    result_value: float = Field(..., description="The numeric value of the measurement")
    result_units: Optional[str] = Field(
        None, description="Units of measurement where applicable"
    )


class CategoricalResult(BaseModel):
    result_name: str = Field(
        ...,
        description="Classification system (e.g., 'Pathogenicity', 'Expression Level' etc)",
    )
    result_value: str = Field(
        ..., description="Assigned category (e.g., 'Pathogenic', 'Positive', 'High')"
    )


class BiomarkerTestResult(BaseModel):
    proband: Optional[str] = Field(..., description="Patient or relatives")
    test_type: TestType = Field(
        ..., description="The category of genomic test performed"
    )
    other_test_type: Optional[str] = Field(
        None, description="Name of test type if 'Other' is selected"
    )
    test_methodology: Optional[str] = Field(
        None, description="Technical description of the test methodology"
    )
    sample_origin: Optional[str] = Field(
        None, description="Anatomical source of the sample tested"
    )
    result_entity_type: Optional[ResultEntityType] = Field(
        None, description="The type of the primary entity being reported"
    )
    result_entity: Optional[str] = Field(
        None, description="The name of the primary entity being reported"
    )
    gene_nomenclature: Optional[List[str]] = Field(
        default_factory=None,
        description="Gene nomenclature used",
    )
    result_region: Optional[str] = Field(
        None, description="The specific region or variant within entity being reported"
    )
    result_status: ResultStatus = Field(
        ..., description="Overall status of the test result"
    )
    result_description: str = Field(
        ..., description="Full textual description of the findings as reported"
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
        description="Comprehensive interpretation of test significance for the patient's condition",
    )
    overall_recommendations: str = Field(
        ...,
        description="Consolidated recommendations based on the complete panel of results",
    )


class Metadata(BaseModel):
    version: str = Field(..., description="Schema version")
    schema_guidelines: List[str] = Field(
        default_factory=list, description="Guidelines for schema usage"
    )
    TO_DO: List[str] = Field(
        default_factory=list, description="Pending tasks for schema improvement"
    )


class GenomicTestReport(BaseModel):
    # metadata: Metadata = Field(..., description="Metadata about the schema")
    clinical_context: ClinicalContext = Field(
        ..., description="Clinical context for the test"
    )
    biomarker_test_results: Optional[List[BiomarkerTestResult]] = Field(
        ..., description="Results of individual biomarker tests performed"
    )
    clinical_outcome: ClinicalOutcome = Field(
        ..., description="Overall interpretation and recommendations"
    )
