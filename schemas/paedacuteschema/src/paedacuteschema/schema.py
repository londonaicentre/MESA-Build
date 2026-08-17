from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

# TYPES & ENUMS

Year = Annotated[int, Field(ge=1900, le=2100)]
Month = Annotated[int, Field(ge=1, le=12)]

class AcuteSignType(str, Enum):

    # RESPIRATORY — WORK OF BREATHING
    TRACHEAL_TUG = "tracheal_tug"
    CHEST_RECESSION = "chest_recession"
    NASAL_FLARING = "nasal_flaring"
    GRUNTING = "grunting"
    HEAD_BOBBING = "head_bobbing"
    APNOEA = "apnoea"
    AUDIBLE_WHEEZE = "audible_wheeze"
    STRIDOR = "stridor"
    WORK_OF_BREATHING = "work_of_breathing"
    SHORTNESS_OF_BREATH = "shortness_of_breath"

    # PERFUSION & SKIN
    PALLOR = "pallor"
    MOTTLING = "mottling"
    PERIPHERAL_CYANOSIS = "peripheral_cyanosis"
    CENTRAL_CYANOSIS = "central_cyanosis"
    PROLONGED_CAPILLARY_REFILL = "prolonged_capillary_refill"
    COLD_PERIPHERIES = "cold_peripheries"
    PETECHIAE = "petechiae"
    PURPURA = "purpura"
    BLANCHING_RASH = "blanching_rash"
    NON_BLANCHING_RASH = "non_blanching_rash"
    JAUNDICE = "jaundice"

    # NEUROLOGICAL & MOTOR
    SEIZURE = "seizure"
    BULGING_FONTANELLE = "bulging_fontanelle"
    NECK_STIFFNESS = "neck_stiffness"
    PHOTOPHOBIA = "photophobia"
    DROWSINESS = "drowsiness"
    REDUCED_CONSCIOUSNESS = "reduced_consciousness"
    CONFUSION = "confusion"
    LETHARGY = "lethargy"
    FLOPPINESS = "floppiness"
    INABILITY_TO_STAND_OR_WALK = "inability_to_stand_or_walk"

    # CRYING & VOCALISATION
    HIGH_PITCHED_CRY = "high_pitched_cry"
    WEAK_CRY = "weak_cry"
    INCONSOLABLE_CRYING = "inconsolable_crying"

    # BEHAVIOUR & INTERACTION
    ABSENT_SOCIAL_INTERACTION = "absent_social_interaction"
    ABSENT_SOCIAL_SMILE = "absent_social_smile"
    REDUCED_PLAY = "reduced_play"
    IRRITABILITY = "irritability"

    # FEEDING, HYDRATION & OUTPUT
    POOR_FEEDING = "poor_feeding"
    UNABLE_TO_DRINK = "unable_to_drink"
    REDUCED_URINE_OUTPUT = "reduced_urine_output"

    # MUSCULOSKELETAL
    JOINT_SWELLING = "joint_swelling"
    JOINT_PAIN = "joint_pain"
    LIMB_PAIN = "limb_pain"
    REFUSAL_TO_WEIGHT_BEAR = "refusal_to_weight_bear"

    # CARER & CLINICIAN CONCERN
    PARENT_REPORTS_NOT_THEMSELVES = "parent_reports_not_themselves"
    CLINICIAN_DOCUMENTS_UNWELL = "clinician_documents_unwell"

    # INFECTION RISK
    HISTORY_OF_FEVER = "history_of_fever"
    RIGORS = "rigors"
    NIGHT_SWEATS = "night_sweats"
    RUNNY_NOSE = "runny_nose"
    COUGH = "cough"
    SORE_THROAT = "sore_throat"
    EARACHE = "earache"
    RASH = "rash"
    WOUND_INFECTION = "wound_infection"
    ABDOMINAL_PAIN = "abdominal_pain"
    ABDOMINAL_DISTENSION = "abdominal_distension"
    DYSURIA = "dysuria"
    VOMITING = "vomiting"
    ALTERED_BOWEL_HABIT = "altered_bowel_habit"
    LYMPH_NODE_SWELLING = "lymph_node_swelling"
    


class Assertion(str, Enum):
    """
    Is the entity asserted as true?
    """
    POSITIVE = "positive"
    # Entity is affirmed as present/true/occurred or has been clinically diagnosed

    NEGATED = "negated"
    # Explicitly ruled out or denied (e.g., "no pain", "denies fever")

    HYPOTHETICAL = "hypothetical"
    # Uncertain, queried, or differential (e.g., "?low UO", "consider", "cannot exclude")

    AMBIGUOUS = "ambiguous"
    # Truth status unclear from context - **use as escape valve if unsure!**


# MODELS

class Entity(BaseModel):
    entity: AcuteSignType = Field(description="The acute sign identified")
    entity_source_text: str = Field(
        description="Short verbatim text that triggered this sign"
    )
    assertion: Assertion = Field(description="Assertion status of the sign")
    is_patient: bool = Field(
        description="True if the sign applies to the child being triaged (not a sibling, parent, or other person)"
    )
    is_active: bool = Field(
        description="True if the sign is part of the current presentation (not purely historical/background)"
    )
    status_source_text: List[str] = Field(
        default_factory=list,
        max_length=4,
        description="1 to 4 short verbatim snippets (each ≤6 words) supporting assertion/is_patient/is_active. Leave empty if none."
    )
    year: Optional[Year] = Field(
        None, description="Year of onset, only if explicitly stated"
    )
    month: Optional[Month] = Field(
        None, description="Month of onset, only if explicitly stated"
    )


class DocumentContent(BaseModel):
    doc_type: str = Field(
        description="Document type (e.g., triage note, nursing assessment, not clinical)"
    )
    doc_summary: str = Field(
        description="Brief summary of the presentation preserving key concepts"
    )


class PaedAcuteSchemaModel(BaseModel):
    is_clinical_document: bool = Field(
        description="True if document contains patient clinical information"
    )
    extraction_reasoning: str = Field(
        description="In <200 words: (1) flag ambiguities and edge cases; (2) note any assertion/applicability uncertainties; (3) confirm omissions — do not infer, prioritise precision over recall."
    )
    document_content: DocumentContent
    entities: Optional[List[Entity]] = None