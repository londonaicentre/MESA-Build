from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

# TYPES & ENUMS

Year = Annotated[int, Field(ge=1900, le=2100)]
Month = Annotated[int, Field(ge=1, le=12)]

class PaedSymptom(str, Enum):

    # RESPIRATORY — WORK OF BREATHING
    TRACHEAL_TUG = "tracheal tug"
    CHEST_RECESSION = "chest recession, intercostal recession, subcostal recession, sternal recession, suprasternal recession, chest indrawing, chest retractions, drawing in"
    NASAL_FLARING = "nasal flaring, flaring nostrils, nares flaring"
    GRUNTING = "grunting, expiratory grunt"
    HEAD_BOBBING = "head bobbing, head nodding"
    PARADOXICAL_BREATHING = "paradoxical breathing, see-saw breathing, abdominal breathing"
    APNOEA = "apnoea, apnea, breathing pauses, stopped breathing, breath holding"
    AUDIBLE_WHEEZE = "audible wheeze, wheeze, wheezing"
    STRIDOR = "stridor, noisy breathing, crowing sound, harsh inspiratory noise"

    # PERFUSION & SKIN
    PALLOR = "pallor, pale, paleness, white, ashen, grey, washed out"
    MOTTLING = "mottling, mottled, marbled skin, blotchy, lace-like pattern"
    CYANOSIS = "peripheral cyanosis, blue hands, blue feet, acrocyanosis, peripheral blueness, central cyanosis, blue lips, blue tongue, central blueness"
    PROLONGED_CAPILLARY_REFILL = "prolonged capillary refill, slow CRT, CRT >2s, cap refill delayed, poor perfusion"
    COLD_PERIPHERIES = "cold peripheries, cold hands, cold feet, cool extremities, shut down peripherally"
    PETECHIAE = "petechiae, petechial rash, pinprick rash"
    PURPURA = "purpura, purpuric rash, non-blanching rash, blood spots"
    BLANCHING_RASH = "blanching rash, blanching spots, viral rash, erythematous rash"
    NON_BLANCHING_RASH = "non-blanching rash, non blanching rash, meningococcal rash"
    JAUNDICE = "jaundice, jaundiced, yellow, yellowing, icterus, icteric"

    # NEUROLOGICAL & MOTOR
    SEIZURE = "seizure, fit, convulsion, twitching, jerking, epileptic episode, shaking episode"
    DECORTICATE_POSTURING = "decorticate posturing, abnormal flexion"
    DECEREBRATE_POSTURING = "decerebrate posturing, abnormal extension"
    OPISTHOTONOS = "opisthotonos, arched back, back arching, hyperextension of back"
    BULGING_FONTANELLE = "bulging fontanelle, tense fontanelle, full fontanelle, raised fontanelle"
    NECK_STIFFNESS = "neck stiffness, stiff neck, nuchal rigidity, neck rigidity, meningism"
    KERNIGS_SIGN = "kernigs sign, Kernig positive"
    BRUDZINSKIS_SIGN = "brudzinskis sign, Brudzinski positive"
    PHOTOPHOBIA = "photophobia, light sensitive, sensitivity to light, avoids light, eyes closed in light"
    PHONOPHOBIA = "phonophobia, noise sensitive, sensitivity to noise, distressed by sound"
    REDUCED_CONSCIOUSNESS = "reduced consciousness, decreased GCS, unresponsive, drowsy, hard to rouse, difficult to wake, not responding"
    CONFUSION = "confusion, confused, disoriented, not making sense, muddled, agitated, delirious"
    LETHARGY = "lethargy, lethargic, sluggish, flat, quiet, very tired, no energy"
    FLOPPINESS = "floppiness, floppy, hypotonic, low tone, rag doll"
    INABILITY_TO_STAND_OR_WALK = "inability to stand or walk, can't stand, can't walk, won't stand"

    # CRYING & VOCALISATION
    HIGH_PITCHED_CRY = "high pitched cry, high-pitched cry, neurological cry, shrill cry, unusual cry"
    WEAK_CRY = "weak cry, feeble cry, absent cry, no cry"
    INCONSOLABLE_CRYING = "inconsolable crying, inconsolable, won't settle, cannot be comforted, persistent crying, constant crying"

    # BEHAVIOUR & INTERACTION
    ABSENT_SOCIAL_INTERACTION = "absent social interaction, not engaging, no eye contact, ignoring parent, blank stare, vacant"
    ABSENT_SOCIAL_SMILE = "absent social smile, not smiling, no smile"
    REDUCED_PLAY = "reduced play, not playing, disinterested, not interested in toys"

    # FEEDING, HYDRATION & OUTPUT
    POOR_FEEDING = "poor feeding, not feeding, reduced feeds, feeding poorly, not taking feeds, refusing breast, refusing bottle, off feeds"
    UNABLE_TO_DRINK = "unable to drink, not swallowing, refusing fluids, dribbling, can't swallow"
    REDUCED_URINE_OUTPUT = "reduced urine output, oliguria, reduced wet nappies, fewer wet nappies, dry nappies, not passed urine, not weed, no urine"

    # MUSCULOSKELETAL
    JOINT_SWELLING = "joint swelling, swollen joint, swollen knee, swollen hip, hot joint, erythematous joint"
    JOINT_PAIN = "joint pain, arthralgia, painful joint, sore joint"
    LIMB_PAIN = "limb pain, leg pain, arm pain, sore limb, limping, limp"
    REFUSAL_TO_WEIGHT_BEAR = "refusal to weight bear, won't weight bear, not weight bearing, won't walk, refuses to walk"

    # CARER & CLINICIAN CONCERN
    PARENT_REPORTS_NOT_THEMSELVES = "parent reports not themselves, not right, not themselves, something wrong, not normal, parent concerned, mum concerned, dad concerned, carer concerned"
    CLINICIAN_DOCUMENTS_UNWELL = "clinician documents unwell, looks unwell, appears unwell, toxic, very sick, critically ill, concerning appearance"

class Assertion(str, Enum):
    """
    Is the entity asserted as true?
    """
    POSITIVE = "positive"
    # Entity is affirmed as present/true/occurred or has been clinically diagnosed

    NEGATED = "negated"
    # Explicitly ruled out or denied (e.g., "no chest pain", "denies fever")

    HYPOTHETICAL = "hypothetical"
    # Uncertain, queried, or differential (e.g., "?PE", "consider", "cannot exclude")

    AMBIGUOUS = "ambiguous"
    # Truth status unclear from context - **use as escape valve if unsure!**


class Subject(str, Enum):
    """
    To whom does the entity apply?
    """
    PATIENT = "patient"
    # Directly applies to the patient

    FAMILY_HISTORY = "family_history"
    # Pertains to a family member (e.g., "FHx breast cancer")

    MOTHER_OR_CHILD_OR_FOETUS = "mother_or_child_or_foetus"
    # Applies to mother or child referances in maternal/newborn notes
    # If mother's note, then this is used for child/foetal concepts
    # If child's note, then this is used for maternal references

    OTHER = "other"
    # Another individual (e.g., "donor HIV positive", "partner has chlamydia")

    AMBIGUOUS = "ambiguous"
    # Subject unclear from context - **use as escape valve if unsure!**


class Relevance(str, Enum):
    """
    When is the entity actively relevant, relative to the current document?
    """
    ACTIVE = "active"
    # Currently relevant as present, ongoing care, or under continued management
    # (e.g. acute presentation being actively managed, active symptom or observation, on-going chronic conditon)

    RECENT = "recent"
    # No longer active but part of key series of events within, or related to this episode
    # (e.g. "DKA on admission, now corrected", "presenting chest pain, now settled", "recent discharge for sepsis with MOF", "initial fever, now afebrile")

    HISTORICAL = "historical"
    # Occurred or resolved in the past, clear separation from current episode
    # (e.g. "PMHx: MI 2019", "previous appendicectomy", "cancer in remission", "childhood asthma, no longer treated")

    PLANNED = "planned"
    # Scheduled or intended for future
    # (e.g. "listed for CABG", "plan for endoscopy")

    AMBIGUOUS = "ambiguous"
    # Temporal relevance unclear from context - **use as escape valve if unsure!**

class Characterisation(BaseModel):
    summary: str = Field(
        description="Normalised summary of severity, progress, or other qualifiers (e.g., 'stable and improving')"
    )
    source_text: List[str] = Field(
        default_factory=list,
        max_length=4,
        description="1 to 4 short verbatim text snippets (each ≤6 words) that triggered characterisation. Leave empty if no relevant evidence"
    )

# MODELS

class Entity(BaseModel):
    entity: EntityType = Field(description="Entity type")
    entity_source_text: str = Field(
        description="Short verbatim text that triggered this entity"
    )
    is_exact_or_umbrella_match: bool = Field(
        description="True if matched entity is an exact or umbrella term for concept. When in doubt, False."
    )
    is_primary_entity: bool = Field(
        description= "True ONLY for the 1-3 entities that are the primary reason for this encounter. This is the chief complaint or main diagnosis. When in doubt, mark False."
    )
    assertion: Assertion = Field(description="Is the entity asserted as true?")
    subject: Subject = Field(description="To whom does the entity apply?")
    relevance: Relevance = Field(description="When is the entity relevant?")
    status_source_text: List[str] = Field(
        default_factory=list,
        max_length=4,
        description="1 to 4 short verbatim text snippets (each ≤6 words) that triggered assertion/subject/activity. Leave empty if no relevant evidence."
    )
    characterisation: Optional[Characterisation] = Field(
        None,
        description="Severity, progress, or other clinical qualifiers"
    )
    year: Optional[Year] = Field(
        None, description="Year of diagnosis or occurrence, only if explicitly stated"
    )
    month: Optional[Month] = Field(
        None, description="Month of diagnosis or occurrence, only if explicitly stated"
    )


class DocumentContent(BaseModel):
    doc_type: str = Field(
        description="Document type (e.g., discharge summary, clinic letter, not clinical)"
    )
    doc_summary: str = Field(
        description="Brief summary of document content preserving key concepts"
    )
    has_active_malignancy: bool = Field(
        description="True if patient has active cancer documented"
    )


class EntitySchemaModel(BaseModel):
    is_clinical_document: bool = Field(
        description="True if document contains patient clinical information"
    )
    extraction_reasoning: str = Field(
        description="In <200 words, think about your approach: (1) Flag ambiguities and edge cases (2) Think through exact/umbrella matching strategy and note status ambiguities; (3) Confirm omissions, remembering that we must not infer, and must prioritise precision over recall."
    )
    document_content: DocumentContent
    entities: Optional[List[Entity]] = None