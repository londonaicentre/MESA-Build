from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

# TYPES & ENUMS

Year = Annotated[int, Field(ge=1900, le=2100)]
Month = Annotated[int, Field(ge=1, le=12)]


class TopographyType(str, Enum):
    UNKNOWN_PRIMARY = "unknown_primary"  # explicitly stated that primary is unknown
    OTHER = "other"  # where unable to fit into any categories below

    # Any haematological or lymphatic
    HAEMATOLOGICAL = "haematological"

    # Respiratory
    LUNG = "lung"
    PLEURA = "pleura"
    OTHER_RESPIRATORY = "other_respiratory"

    # GI tract
    OESOPHAGUS = "oesophagus"
    STOMACH = "stomach"
    SMALL_INTESTINE = "small_intestine"
    COLON = "colon"
    RECTUM = "rectum"
    PANCREAS = "pancreas"
    LIVER = "liver"
    GALLBLADDER = "gallbladder"
    BILE_DUCT = "bile_duct"
    OTHER_GI = "other_gi"

    # GU
    KIDNEY = "kidney"
    BLADDER = "bladder"
    PROSTATE = "prostate"
    TESTIS = "testis"
    OTHER_GU = "other_gu"

    # Female
    BREAST = "breast"
    CERVIX = "cervix"
    UTERUS = "uterus"
    OVARY = "ovary"
    OTHER_GYNAE = "other_gynae"

    # CNS
    BRAIN = "brain"
    SPINAL_CORD = "spinal_cord"
    OTHER_CNS = "other_cns"

    # Head & Neck
    ORAL = "oral"  # any oral cavity
    HYPO_ORO_NASO_PHARYNX = "hypo_oro_naso_pharynx"  # any pharynx
    LARYNX = "larynx"
    SALIVARY_GLAND = "salivary_gland"
    NASAL_CAVITY = "nasal_cavity"
    PARANASAL_SINUS = "paranasal_sinus"

    # Endocrine
    THYROID = "thyroid"
    ADRENAL_GLAND = "adrenal_gland"
    OTHER_ENDOCRINE = "other_endocrine"

    # Skin/soft tissue/MSK
    SKIN = "skin"
    SOFT_TISSUE = "soft_tissue"
    BONE = "bone"


class MorphologyType(str, Enum):
    UNKNOWN_MORPHOLOGY = (
        "unknown_morphology"  # explicitly stated that morphology unknown or uncertain
    )
    OTHER = "other"  # where unable to fit into any categories below

    # Carcinomas
    ADENOCARCINOMA = "adenocarcinoma"
    SQUAMOUS_CELL_CARCINOMA = "squamous_cell_carcinoma"
    UROTHELIAL_CARCINOMA = "urothelial_carcinoma"
    RENAL_CELL_CARCINOMA = "renal_cell_carcinoma"
    HEPATOCELLULAR_CARCINOMA = "hepatocellular_carcinoma"
    SMALL_CELL_CARCINOMA = "small_cell_carcinoma"
    NON_SMALL_CELL_CARCINOMA = "non_small_cell_carcinoma"
    CARCINOMA_OTHER = "carcinoma_other"

    # Mesothelioma
    MESOTHELIOMA = "mesothelioma"

    # Melanocytic
    MELANOMA = "melanoma"

    # Neuroendocrine
    NEUROENDOCRINE = "neuroendocrine"
    CARCINOID = "carcinoid"

    # Sarcoma
    SARCOMA_NOS = "sarcoma_nos"
    GIST = "gastrointestinal_stromal_tumour"
    OSTEOSARCOMA = "osteosarcoma"
    CHONDROSARCOMA = "chondrosarcoma"
    EWING_SARCOMA = "ewing_sarcoma"
    RHABDOMYOSARCOMA = "rhabdomyosarcoma"
    KAPOSI_SARCOMA = "kaposi_sarcoma"
    SOFT_TISSUE_SARCOMA = "soft_tissue_sarcoma"

    # Haematological
    ACUTE_LYMPHOBLASTIC_LEUKAEMIA = "acute_lymphoblastic_leukaemia"
    ACUTE_MYELOID_LEUKAEMIA = "acute_myeloid_leukaemia"
    CHRONIC_LYMPHOCYTIC_LEUKAEMIA = "chronic_lymphocytic_leukaemia"
    CHRONIC_MYELOID_LEUKAEMIA = "chronic_myeloid_leukaemia"
    HODGKIN_LYMPHOMA = "hodgkin_lymphoma"
    NON_HODGKIN_LYMPHOMA = "non_hodgkin_lymphoma"
    MULTIPLE_MYELOMA = "multiple_myeloma"
    LEUKAEMIA_OTHER = "leukaemia_other"
    MYELODYSPLASTIC = "myelodysplastic"

    # CNS
    GLIOBLASTOMA = "glioblastoma"
    ASTROCYTOMA = "astrocytoma"
    OLIGODENDROGLIOMA = "oligodendroglioma"
    MENINGIOMA = "meningioma"

    # Germ cell
    SEMINOMA = "seminoma"
    TERATOMA = "teratoma"
    CHORIOCARCINOMA = "choriocarcinoma"

    # Paediatric embryonal
    WILMS_TUMOUR = "wilms_tumour"
    RETINOBLASTOMA = "retinoblastoma"
    HEPATOBLASTOMA = "hepatoblastoma"


class MolecularBiomarkerType(str, Enum):
    # Other
    OTHER = "other"

    # General
    BRAF = "braf"
    NTRK1 = "ntrk1"
    NTRK2 = "ntrk2"
    NTRK3 = "ntrk3"
    RET = "ret"
    ERBB2_HER2 = "erbb2_her2"
    TP53 = "tp53"

    # DNA repair
    BRCA1 = "brca1"
    BRCA2 = "brca2"
    MLH1 = "mlh1"
    MSH2 = "msh2"
    MSH6 = "msh6"
    PMS2 = "pms2"
    PALB2 = "palb2"
    RAD51 = "rad51"

    # Lung
    EGFR = "egfr"
    ALK = "alk"
    ROS1 = "ros1"
    MET = "met"
    KRAS = "kras"

    # Colorectal
    NRAS = "nras"

    # Breast
    PIK3CA = "pik3ca"
    ESR1_ER = "esr1_er"
    PGR_PR = "pgr_pr"
    KI67 = "ki_67"

    # GIST/Melanoma
    KIT = "kit"
    PDGFRA = "pdgfra"

    # Bladder/Cholangiocarcinoma
    FGFR1 = "fgfr1"
    FGFR2 = "fgfr2"
    FGFR3 = "fgfr3"

    # Glioma/Cholangiocarcinoma
    IDH1 = "idh1"
    IDH2 = "idh2"

    # Other solid tumour
    PDL1 = "pdl1"
    NF1 = "nf1"
    NF2 = "nf2"

    # Neuro-oncology
    MGMT = "mgmt"

    # Haematological
    NPM1 = "npm1"
    FLT3 = "flt3"
    BCR_ABL1 = "bcr_abl1"
    JAK2 = "jak2"
    BCL2 = "bcl2"
    MYC = "myc"


class BiomarkerStatus(str, Enum):
    ALTERED = (
        "altered"  # any alteration or high expression recorded (e.g. BRAF V600E, PR 8+)
    )
    NEGATIVE = (
        "negative"  # explicit recording of negativity or normality (e.g. HER -ve)
    )
    EQUIVOCAL = "equivocal"  # Uncertain or ambiguous result
    HYPOTHETICAL = (
        "hypothetical"  # biomarker is postulated, no result (e.g. awaiting PDL-1)
    )


class MSIStatus(str, Enum):
    MSI_HIGH = "msi_high"
    MS_STABLE = "ms_stable"


class TMBStatus(str, Enum):
    TMB_HIGH = "tmb_high"
    TMB_LOW = "tmb_low"
    TMB_INTERMEDIATE = "tmb_intermediate"


class PatientFindingStatus(str, Enum):
    IS_PRESENT = "is_present"  # positive findings
    IS_NOT_PRESENT = "is_not_present"  # e.g. patient DENIES symptom, sign NOT found, patient's condition has RESOLVED
    UNCERTAIN = "uncertain"  # e.g. patient may have x


class SpreadType(str, Enum):
    OTHER = "other"
    LYMPH_NODE = "lymph_node"
    LIVER = "liver"
    LUNG = "lung"
    SPINE = "spine"
    OTHER_BONE = "other_bone"  # bone but not spine
    BRAIN = "brain"
    OTHER_CNS = "other_cns"  # cns but not brain
    ADRENAL = "adrenal"
    KIDNEY = "kidney"
    PLEURA = "pleura"
    PERITONEUM = "peritoneum"
    SKIN = "skin"
    PANCREAS = "pancreas"
    SPLEEN = "spleen"
    OVARY = "ovary"
    TESTIS = "testis"
    THYROID = "thyroid"
    STOMACH = "stomach"
    BOWEL = "bowel"
    BLADDER = "bladder"
    PROSTATE = "prostate"
    BREAST = "breast"
    HEAD_AND_NECK = "head_and_neck"  # head and neck region


class CancerScoreName(str, Enum):
    PATHOLOGICAL_GRADE = "pathological_grade"
    GLEASON = "gleason"
    FIGO = "figo"
    DUKES = "dukes"
    BRESLOW = "breslow"
    CLARK = "clark"
    BINET = "binet"
    CHILD_PUGH = "child_pugh"
    OTHER = "other"  # any scores outside of this set


class TimelineEventType(str, Enum):
    HAD_SYSTEMIC_OR_RADIOTHERAPY_TREATMENT = "had_systemic_or_radiotherapy_treatment"  # for new treatments (or switch onto a treatment)
    HAD_SURGICAL_TREATMENT_PERFORMED = "had_surgical_treatment_performed"
    EXPERIENCED_TOXICITY_OR_COMPLICATION_RELATED_TO_TREATMENT = (
        "experienced_toxicity_or_complication_related_to_treatment"
    )
    EXPERIENCED_TREATMENT_REDUCTION_OR_STOP = "experienced_treatment_reduction_or_stop"  # where a current treatment was stopped or dose reduced, e.g. due to negative effects
    CONSIDERED_FOR_CLINICAL_TRIAL = "considered_for_clinical_trial"  # any discussion of patient being under consideration for a trial
    ENROLLED_TO_CLINICAL_TRIAL = "enrolled_to_clinical_trial"  # only where patient is confirmed to be enrolled in any trial
    POSITIVE_TREATMENT_RESPONSE_ON_ASSESSMENT = "positive_treatment_response_on_assessment"  # there has been explicit assessment of response to treatment, and patient has had a positive response (e.g. on imaging or on recist)
    EVIDENCE_OF_METASTATIC_PROGRESSION = "evidence_of_metastatic_progression"  # explicit confirmation of new nodal or distant spread
    RADIOLOGY_EVIDENCE_OF_DISEASE_PROGRESSION = "radiology_evidence_of_disease_progression"  # explicit evidence of any disease progression on imaging
    EXPERIENCED_DISEASE_REMISSION = (
        "experienced_disease_remission"  # explicit mention that patient in remission
    )
    EXPERIENCED_DISEASE_RECURRENCE = "experienced_disease_recurrence"  # where primary cancer is explicitly described as recurring or relapsing
    PATIENT_DIED = "patient_died"


class PatientFindingType(str, Enum):
    COMORBIDITY_FINDING = "comorbidity_finding"  # concurrent diagnosed condition
    SOCIAL_OR_FAMILY_FINDING = "social_or_family_finding"  # e.g. social housing or care package, smoking hx, alcohol hx, family hx
    SYMPTOM_FINDING = "symptom_finding"  # patient has experienced a symptom e.g. pain, shortness of breath, lost weight
    PHYSICAL_EXAMINATION_FINDING = "physical_examination_finding"  # clinician observes a physical finding e.g. abdominal distension, cachexia
    FUNCTIONAL_FINDING = "functional_finding"  # comment on patient physical function, e.g. performance status, able to walk
    MENTAL_STATE_FINDING = "mental_state_finding"  # comment on patient mental state / wellbeing e.g. feels positive, deeply upset


class FuturePlanType(str, Enum):
    PLANNED_SYSTEMIC_OR_RADIOTHERAPY_TREATMENT = (
        "planned_systemic_or_radiotherapy_treatment"
    )
    PLANNED_SURGERY_TREATMENT = "planned_surgery_treatment"
    PLANNED_INVESTIGATION = "planned_investigation"
    PLANNED_CLINICAL_TRIAL_INVOLVEMENT = "planned_clinical_trial_involvement"


# BLOCKS


class PrimaryCancerFacts(BaseModel):
    topography: TopographyType = Field(
        description="Most suitable anatomical site of primary cancer. Use OTHER where there is ambiguity, or no suitable found in enum."
    )
    topography_name_desc: Optional[str] = Field(
        None, description="Name of anatomical site as described in clinical text"
    )
    morphology: Optional[MorphologyType] = Field(
        None,
        description="Most suitable histological classification of primary cancer. Use OTHER where there is ambiguity, or no suitable found in enum.",
    )
    morphology_name_desc: Optional[str] = Field(
        None,
        description="Name of histological classification as described in clinical text",
    )
    is_recurrence: bool = Field(
        description="TRUE if the primary cancer is explicitly described as a recurrence or relapse of a previous cancer"
    )
    diagnosis_year: Optional[Year] = Field(
        None, description="Year of initial diagnosis"
    )
    diagnosis_month: Optional[Month] = Field(
        None, description="Month of initial diagnosis"
    )
    tnm_stage: Optional[str] = Field(
        None,
        description="Confirmed, latest, TNM staging for main cancer (e.g. T2N1bM0)",
    )
    numeric_group_stage: Optional[str] = Field(
        None,
        description="Confirmed group staging for main cancer using Roman numerals (e.g. I, IIa, IV etc)",
    )


class PrimaryCancerScore(BaseModel):
    score: CancerScoreName = Field(
        description="Name of a single specialist staging or grading measurement. Use OTHER where ambiguity or score not in enum."
    )
    score_name_desc: str = Field(
        description="Name of a staging or grading measurement as described in clinical text"
    )
    score_value: Optional[str] = Field(None, description="Value if score is quantified")
    score_value_desc: Optional[str] = Field(
        None, description="Direct extract of score value as described in clinical text"
    )


class PrimaryCancerSpread(BaseModel):
    spread_type: SpreadType = Field(
        description="Most suitable type of spread. Use OTHER where ambiguity or no suitable found."
    )
    spread_site_desc: str = Field(
        description="Name of specific metastastic site or node as described in clinical text"
    )


class MolecularBiomarkerProfile(BaseModel):
    biomarker: MolecularBiomarkerType = Field(
        description="Gene or protein biomarker. Use OTHER if specific biomarker not listed."
    )
    biomarker_name_desc: Optional[str] = Field(
        None, description="Name of biomarker as described in clinical text"
    )
    biomarker_status: BiomarkerStatus = Field(description="Crude status of marker")
    biomarker_alteration: Optional[str] = Field(
        None, description="Description of specific alteration (e.g. V600E, G12C)"
    )
    biomarker_expression_level: Optional[str] = Field(
        None,
        description="Description of expression level for protein markers if exists (e.g. 3+, 90% positive, 4/8)",
    )
    biomarker_vaf_numeric_value: Optional[float] = Field(
        None, description="Numeric value for variant allele frequency in %"
    )
    biomarker_vaf_desc: Optional[str] = Field(
        None, description="Direct extract of descriptive text for biomarker VAF"
    )
    biomarker_desc: Optional[str] = Field(
        None,
        description="Direct extract of descriptive text that fully describes biomarker and status, including alterations and numeric values",
    )


class PrimaryCancerTimelineEvent(BaseModel):
    event_type: TimelineEventType = Field(description="Select type of event")
    event_year: Optional[Year] = Field(None, description="Year of event")
    event_month: Optional[Month] = Field(None, description="Month of event")
    event_summary: str = Field(
        description="Short summary for specific event with essential facts"
    )
    event_desc: str = Field(description="Direct extract of descriptive text for event")


class PrimaryCancerTumourFacts(BaseModel):
    msi_status: Optional[MSIStatus] = Field(
        None, description="Microsatellite instability status"
    )
    msi_desc: Optional[str] = Field(
        None, description="Direct extract of descriptive text for MSI"
    )
    tmb_status: Optional[TMBStatus] = Field(
        None, description="Tumour mutation burden status"
    )
    tmb_numeric_value: Optional[float] = Field(
        None, description="TMB numeric value in mutations/Mb if provided"
    )
    tmb_desc: Optional[str] = Field(
        None, description="Direct extract of descriptive text for TMB"
    )
    molecular_biomarker_profiles: Optional[List[MolecularBiomarkerProfile]] = Field(
        None, description="Genomic alterations and protein expression findings"
    )


class PrimaryCancer(BaseModel):
    primary_cancer_facts: PrimaryCancerFacts = Field(
        description="Main facts about primary cancer"
    )
    primary_cancer_tumour_facts: Optional[PrimaryCancerTumourFacts] = Field(
        None, description="Molecular and genomic characteristics of the tumour"
    )
    primary_cancer_scores: Optional[List[PrimaryCancerScore]] = Field(
        None, description="Specialist cancer staging or grading scores"
    )
    primary_cancer_spread: Optional[List[PrimaryCancerSpread]] = Field(
        None, description="Locations of cancer spread - confirmed sites only"
    )
    primary_cancer_timeline_events: Optional[List[PrimaryCancerTimelineEvent]] = Field(
        None, description="Timeline of historical events"
    )


class PerformanceStatus(BaseModel):
    ps_scale: Optional[str] = Field(
        None,
        description="Scale used for performance status if given, e.g. ECOG, Karnofsky",
    )
    ps_score_value: str = Field(description="Score value (e.g. '0', '1', '70', '100')")
    ps_desc: Optional[str] = Field(
        None, description="Direct extract of descriptive text for performance status"
    )


class OtherCancerFacts(BaseModel):
    other_topography: TopographyType = Field(
        description="Confirmed topography of historical or concurrent cancer"
    )
    other_topography_name_desc: Optional[str] = Field(
        None, description="Name of anatomical site as described in clinical text"
    )
    other_morphology: Optional[MorphologyType] = Field(
        None, description="Confirmed morphology of historical or concurrent cancer"
    )
    other_morphology_name_desc: Optional[str] = Field(
        None,
        description="Name of histological classification as described in clinical text",
    )
    other_diagnosis_year: Optional[Year] = Field(
        None, description="Year of initial diagnosis"
    )
    other_diagnosis_month: Optional[Month] = Field(
        None, description="Month of initial diagnosis"
    )
    other_latest_situation_summary: Optional[str] = Field(
        None, description="Short summary of latest situation for this cancer"
    )


class PatientFinding(BaseModel):
    patient_finding_type: PatientFindingType = Field(
        description="Type of patient finding"
    )
    patient_finding_name_desc: str = Field(
        description="Name of specific finding as described in clinical text"
    )
    patient_finding_desc: str = Field(
        description="Direct extract of descriptive text for the finding"
    )
    patient_finding_status: PatientFindingStatus = Field(
        description="Whether the specific finding is present, is not present, or is uncertain/hypothetical"
    )


class FuturePlan(BaseModel):
    future_plan_type: FuturePlanType = Field(description="Type of plan")
    future_plan_summary: str = Field(
        description="Short summary of future plan with essential facts only"
    )
    future_plan_desc: str = Field(
        description="Direct extract of descriptive text for plan"
    )


class ContextSummary(BaseModel):
    doc_context: str = Field(
        description="Short description of document context (e.g. MDM note, clinic letter, genomics report)"
    )
    doc_summary: str = Field(
        description="Short summary of current cancer treatment status with focus on phase of care"
    )


class OncologyModel(BaseModel):
    document_has_primary_cancer_flag: bool = Field(
        description="TRUE if the document concerns a patient and a cancer diagnosis"
    )
    primary_cancer_confirmed_flag: bool = Field(
        description="TRUE = there is confirmed primary cancer diagnosis, FALSE = hypothetical/under investigation or not cancer"
    )
    primary_cancer: Optional[PrimaryCancer] = None
    performance_status: Optional[PerformanceStatus] = Field(
        None, description="Most recent performance status"
    )
    other_cancers: Optional[List[OtherCancerFacts]] = None
    patient_findings: Optional[List[PatientFinding]] = None
    future_plans: Optional[List[FuturePlan]] = None
    context_summary: Optional[ContextSummary] = None
