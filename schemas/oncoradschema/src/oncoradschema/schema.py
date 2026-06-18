from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ENUMS

class Modality(str, Enum):
    CT = "ct"
    MRI = "mri"
    PET_CT = "pet_ct"
    PET_MRI = "pet_mri"
    SPECT = "spect"
    SPECT_CT = "spect_ct"
    ULTRASOUND = "ultrasound"
    XRAY = "xray"
    FLUOROSCOPY = "fluoroscopy"
    MAMMOGRAPHY = "mammography"
    DEXA = "dexa"
    OTHER = "other"


class Contrast(str, Enum):
    WITHOUT = "without"
    WITH_CONTRAST = "with_contrast" # single phase
    DUAL_PHASE = "dual_phase"
    TRIPLE_PHASE = "triple_phase"
    NOT_APPLICABLE = "not_applicable"


class Region(str, Enum):
    HEAD = "head"
    NECK = "neck"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    SPINE = "spine"
    UPPER_LIMB = "upper_limb"
    LOWER_LIMB = "lower_limb"
    WHOLE_BODY = "whole_body"
    HEART = "heart"
    BREAST = "breast"
    VESSEL = "vessel"
    OTHER = "other"


class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    NOT_APPLICABLE = "not_applicable"


class ComparativeChange(str, Enum):
    NEW = "new"
    PROGRESSIVE = "progressive"
    STABLE = "stable"
    IMPROVING = "improving"
    RESOLVED = "resolved"
    MIXED = "mixed"
    INDETERMINATE = "indeterminate"


class RECISTResponse(str, Enum):
    """
    RECIST 1.1 treatment response categories on oncological follow-up imaging.
    """
    COMPLETE_RESPONSE = "complete_response" # disappearance of all target lesions
    PARTIAL_RESPONSE = "partial_response" # ≥30% decrease in sum of diameters
    STABLE_DISEASE = "stable_disease" # neither PR nor PD criteria met
    PROGRESSIVE_DISEASE = "progressive_disease" # ≥20% increase or new lesions
    NOT_EVALUABLE = "not_evaluable" # cannot assess — missing, artefact, wrong modality
    NOT_APPLICABLE = "not_applicable" # not a target lesion / not a follow-up scan


class ImageRationale(str, Enum):
    SCREENING = "screening" # asymptomatic population screening
    DIAGNOSIS_STAGING = "diagnosis_staging" # initial workup or staging of known/suspected cancer
    FOLLOW_UP = "follow_up" # any post-treatment imaging: response assessment, surveillance, routine follow-up
    INTERVENTIONAL = "interventional" # image-guided procedure / biopsy
    UNCERTAIN = "uncertain" # rationale not determinable from report
    OTHER = "other" # rationale clear but doesn't fit above (e.g. incidental, pre-op planning)


class ScoringSystem(str, Enum):
    OTHER = "other"
    TNM = "tnm"
    AJCC_STAGE = "ajcc_stage"
    BI_RADS = "bi_rads"
    BI_RADS_DENSITY = "bi_rads_density"
    PI_RADS = "pi_rads"
    BOSNIAK = "bosniak"
    LI_RADS = "li_rads"
    LUNG_RADS = "lung_rads"
    TI_RADS = "ti_rads"
    O_RADS = "o_rads"
    DEAUVILLE = "deauville"
    GLEASON = "gleason"


class LesionCertainty(str, Enum):
    CERTAIN = "certain" # report asserts the lesion as cancer
    UNCERTAIN = "uncertain" # described as possible, probable, query, indeterminate cancer lesion
    UNSPECIFIED = "unspecified" # report does not comment on certainty either way (e.g. "innumerable hepatic lesions")


class LesionNature(str, Enum):
    PRIMARY = "primary" # primary tumour at this site
    METASTASIS = "metastasis" # confirmed or strongly implied secondary
    UNCLEAR = "unclear" # cannot distinguish from report


class LesionMargin(str, Enum):
    """Edge transitions using BI-RADS lexicon (with synonyms)"""
    CIRCUMSCRIBED_WELLDEFINED = "circumscribed_welldefined" # sharp edge (covers "well-defined")
    MICROLOBULATED = "microlobulated" # small undulations along the margin
    OBSCURED = "obscured" # margin hidden by adjacent/superimposed tissue
    INDISTINCT_ILLDEFINED = "indistinct_illdefined" # margin poorly seen; no infiltration implied (covers "ill-defined")
    SPICULATED = "spiculated" # radiating lines from the mass (geometric)
    NOT_DESCRIBED = "not_described" # report does not characterise the margin


class LesionShape(str, Enum):
    """Silhouette descriptors using BI-RADS lexicon"""
    OVAL = "oval" # elliptical
    ROUND = "round" # spherical
    IRREGULAR = "irregular" # non-uniform, no defining pattern
    LOBULATED = "lobulated" # multiple smooth bulges, polycyclic
    NOT_DESCRIBED = "not_described" # report does not characterise the shape


class LesionMorphology(str, Enum):
    """Tissue composition of the lesion"""
    SOLID = "solid"
    CYSTIC = "cystic"
    MIXED_SOLID_CYSTIC = "mixed_solid_cystic"
    GROUND_GLASS_NODULE = "ground_glass_nodule" # Fleischner description
    PART_SOLID_GROUND_GLASS = "part_solid_ground_glass" # Fleischner description (solid + ground-glass)
    NOT_DESCRIBED = "not_described" # report does not characterise composition


class LesionInternalFeature(str, Enum):
    """
    Internal features of the lesion. Multi-valued.
    """
    NECROTIC = "necrotic" # central or regional non-enhancing breakdown
    CALCIFIED = "calcified" # any internal calcification
    HAEMORRHAGIC = "haemorrhagic" # internal haemorrhage / haemorrhagic content
    MUCINOUS = "mucinous" # mucinous content
    FAT_CONTAINING = "fat_containing" # macroscopic fat (e.g. AML, liposarcoma, teratoma)
    SCLEROTIC = "sclerotic" # increased density / blastic (bone lesions)
    LYTIC = "lytic" # reduced density / destructive (bone lesions)
    MIXED_SCLEROTIC_LYTIC = "mixed_sclerotic_lytic" # mixed bone-lesion pattern


class AnatomicalSite(str, Enum):
    OTHER = "other"
    UNKNOWN = "unknown"

    # Thoracic
    LUNG = "lung"
    PLEURA = "pleura"
    MEDIASTINUM = "mediastinum"
    HEART = "heart"
    CHEST_WALL = "chest_wall"

    # Abdominal solid organs
    LIVER = "liver"
    PANCREAS = "pancreas"
    SPLEEN = "spleen"
    GALLBLADDER = "gallbladder"
    BILE_DUCT = "bile_duct"
    KIDNEY = "kidney"
    ADRENAL = "adrenal"

    # GI tract
    OESOPHAGUS = "oesophagus"
    STOMACH = "stomach"
    SMALL_INTESTINE = "small_intestine"
    COLON = "colon"
    RECTUM = "rectum"
    ANUS = "anus"

    # Peritoneal / retroperitoneal
    PERITONEUM = "peritoneum"
    OMENTUM = "omentum"
    MESENTERY = "mesentery"
    RETROPERITONEUM = "retroperitoneum"

    # GU / pelvic
    BLADDER = "bladder"
    PROSTATE = "prostate"
    TESTIS = "testis"
    OVARY = "ovary"
    UTERUS = "uterus"
    CERVIX = "cervix"

    # Breast
    BREAST = "breast"

    # CNS
    BRAIN = "brain"
    MENINGES = "meninges"
    SPINAL_CORD = "spinal_cord"

    # Head & neck
    ORAL_CAVITY = "oral_cavity"
    PHARYNX = "pharynx"
    LARYNX = "larynx"
    SALIVARY_GLAND = "salivary_gland"
    NASAL_CAVITY = "nasal_cavity"
    PARANASAL_SINUS = "paranasal_sinus"
    THYROID = "thyroid"

    # Bone (axial / appendicular)
    BONE_SKULL = "bone_skull"
    BONE_SPINE = "bone_spine"
    BONE_RIBS_STERNUM = "bone_ribs_sternum"
    BONE_PELVIS = "bone_pelvis"
    BONE_UPPER_LIMB = "bone_upper_limb"
    BONE_LOWER_LIMB = "bone_lower_limb"

    # Soft tissue (by body region)
    SOFT_TISSUE_HEAD_NECK = "soft_tissue_head_neck"
    SOFT_TISSUE_THORAX = "soft_tissue_thorax"
    SOFT_TISSUE_ABDOMEN = "soft_tissue_abdomen"
    SOFT_TISSUE_PELVIS = "soft_tissue_pelvis"
    SOFT_TISSUE_UPPER_LIMB = "soft_tissue_upper_limb"
    SOFT_TISSUE_LOWER_LIMB = "soft_tissue_lower_limb"

    # Lymph nodes (by nodal station)
    LYMPH_NODE_CERVICAL = "lymph_node_cervical"
    LYMPH_NODE_SUPRACLAVICULAR = "lymph_node_supraclavicular"
    LYMPH_NODE_AXILLARY = "lymph_node_axillary"
    LYMPH_NODE_MEDIASTINAL = "lymph_node_mediastinal"
    LYMPH_NODE_HILAR = "lymph_node_hilar"
    LYMPH_NODE_ABDOMINAL = "lymph_node_abdominal"
    LYMPH_NODE_RETROPERITONEAL = "lymph_node_retroperitoneal"
    LYMPH_NODE_PELVIC = "lymph_node_pelvic"
    LYMPH_NODE_INGUINAL = "lymph_node_inguinal"

    # Vessels (commonly named in oncology radiology reports)
    VESSEL_AORTA = "vessel_aorta"
    VESSEL_IVC = "vessel_ivc"
    VESSEL_SVC = "vessel_svc"
    VESSEL_PORTAL_VEIN = "vessel_portal_vein"
    VESSEL_HEPATIC_VEIN = "vessel_hepatic_vein"
    VESSEL_HEPATIC_ARTERY = "vessel_hepatic_artery"
    VESSEL_SMA = "vessel_sma"
    VESSEL_SMV = "vessel_smv"
    VESSEL_SPLENIC_VEIN = "vessel_splenic_vein"
    VESSEL_RENAL_VEIN = "vessel_renal_vein"
    VESSEL_RENAL_ARTERY = "vessel_renal_artery"
    VESSEL_PULMONARY_ARTERY = "vessel_pulmonary_artery"
    VESSEL_CAROTID = "vessel_carotid"
    VESSEL_ILIAC = "vessel_iliac"
    VESSEL_OTHER = "vessel_other"


class NonCancerFindingType(str, Enum):
    """
    Mechanism-based taxonomy for clinically-impactful non-cancer findings.
    Anatomical location is captured independently by AnatomicalSite.
    Two concurrent processes at the same site (e.g. pneumonia + parapneumonic effusion)
    are emitted as two separate NonCancerFinding rows.
    """
    OTHER = "other"

    # Vascular
    THROMBUS = "thrombus" # any in-situ clot or embolism (DVT, portal/IVC, PE, mural)
    ANEURYSM = "aneurysm" # aneurysmal dilatation of a vessel
    DISSECTION = "dissection" # intimal/wall dissection of a vessel
    STENOSIS_OCCLUSION = "stenosis_occlusion" # non-embolic narrowing/occlusion

    # Fluid / collection
    EFFUSION = "effusion" # pleural, pericardial, joint, ascites (serous)
    HAEMORRHAGE_HAEMATOMA = "haemorrhage_haematoma" # any bleed or haematoma
    ABSCESS_PUS_COLLECTION = "abscess_pus_collection" # organised infected collection
    CYST_BENIGN_COLLECTION = "cyst_benign_collection" # simple cyst, seroma, lymphocoele

    # Air / gas
    PNEUMOTHORAX_PNEUMOPERITONEUM = "pneumothorax_pneumoperitoneum"

    # Inflammation / infection (tissue-level)
    INFLAMMATION  = "inflammation"    # pneumonia, colitis, cystitis, cholecystitis

    # Obstruction / dilatation of a hollow viscus or duct
    OBSTRUCTION_DILATATION = "obstruction_dilatation"    # bowel obstruction, hydronephrosis, biliary dilatation

    # Calculi
    CALCULUS = "calculus"                                # stones in any duct/cavity

    # Chronic parenchymal change
    FIBROSIS_SCARRING = "fibrosis_scarring"              # ILD, radiation, chronic scar
    STEATOSIS_FATTY_CHANGE = "steatosis_fatty_change"    # hepatic/pancreatic fatty change
    ATROPHY_VOLUME_LOSS = "atrophy_volume_loss"          # parenchymal atrophy, lobar collapse

    # Structural breakdown
    FRACTURE = "fracture"                                # any cortical break
    PERFORATION_RUPTURE = "perforation_rupture"          # bowel perforation, organ rupture
    HERNIATION = "herniation"                            # any hernia, disc herniation


# BLOCKS

class ScanMetadata(BaseModel):
    modality: Modality = Field(description="Imaging modality used")
    is_interventional: bool = Field(
        description="True if the report is for a radiologically guided procedure"
    )
    contrast: Optional[Contrast] = Field(
        None, description="Contrast usage for the scan"
    )
    regions: Optional[List[Region]] = Field(
        None, description="Anatomical regions captured by the scan"
    )


class DiseaseSpecificScore(BaseModel):
    scoring_system: ScoringSystem = Field(
        description="Structured scoring system. Use OTHER if not in enum."
    )
    scoring_system_desc: Optional[str] = Field(
        None, description="Name of scoring system as described in clinical text"
    )
    score_or_stage: str = Field(
        description="Score or stage assigned (e.g. 'T3bN1M0', 'PI-RADS 4', 'Category III', 'C')"
    )
    laterality: Optional[Laterality] = Field(
        None,
        description="Laterality the score applies to, if relevant",
    )
    score_desc: Optional[str] = Field(
        None, description="Direct extract of descriptive text for the score"
    )


class CancerStatus(BaseModel):
    overall_status: Optional[ComparativeChange] = Field(
        None, description="Overall progression status compared to prior imaging"
    )
    recist_response: Optional[RECISTResponse] = Field(
        None, description="RECIST 1.1 response category if reported"
    )
    progression_desc: Optional[str] = Field(
        None, description="Direct extract of descriptive text for overall progression"
    )
    disease_specific_scores: Optional[List[DiseaseSpecificScore]] = Field(
        None,
        description="Structured scoring/staging systems reported (e.g. TNM, BI-RADS, PI-RADS, Bosniak)",
    )


class LesionSize(BaseModel):
    longest_diameter_mm: Optional[float] = Field(
        None,
        ge=0,
        description="Longest reported diameter in mm; use when report specifies a single 'longest' or 'largest' dimension (RECIST-relevant)",
    )
    x_mm: Optional[float] = Field(
        None, ge=0, description="First reported axis in mm"
    )
    y_mm: Optional[float] = Field(
        None, ge=0, description="Second reported axis in mm"
    )
    z_mm: Optional[float] = Field(
        None, ge=0, description="Third reported axis in mm"
    )
    volume_ml: Optional[float] = Field(
        None, ge=0, description="Volume in mL or cc if explicitly reported"
    )


class CancerLesion(BaseModel):
    """A single lesion (or a miliary presentation) present in a given anatomical site"""
    anatomical_site: AnatomicalSite = Field(
        description="Anatomical site of the lesion"
    )
    anatomical_site_desc: Optional[str] = Field(
        None,
        description="Direct extract of descriptive text for the lesion's anatomical site",
    )
    cancer_certainty: LesionCertainty = Field(
        description="Whether the report asserts the lesion as certain, uncertain, or unspecified (use UNSPECIFIED when the report does not comment on certainty)"
    )
    lesion_desc: str = Field(
        description="Direct extract of descriptive text for the lesion",
    )
    change: Optional[ComparativeChange] = Field(
        None, description="Change of this lesion compared to prior imaging"
    )
    laterality: Optional[Laterality] = Field(
        None, description="Laterality of the lesion"
    )
    size: Optional[LesionSize] = Field(None, description="Reported lesion dimensions")
    is_recist_target: bool = Field(
        False, description="True if designated a RECIST target lesion"
    )
    is_largest: bool = Field(
        False, description="True if this is the largest lesion described in the report"
    )
    is_infiltrative: bool = Field(
        False,
        description="True if the lesion is described as actively invading or infiltrating adjacent tissue",
    )
    is_vascular_invasion: bool = Field(
        False,
        description="True if the lesion is described as invading a vessel or forming tumour thrombus (e.g. portal vein tumour thrombus, IVC invasion)",
    )
    is_miliary: bool = Field(
        False,
        description=(
            "True if the lesion is described as miliary (innumerable tiny nodules across an anatomy). "
        ),
    )
    nature: LesionNature = Field(
        description="Whether the lesion is a primary, metastasis, or unclear"
    )
    morphology: LesionMorphology = Field(
        description="Tissue composition of the lesion. Use NOT_DESCRIBED if not characterised."
    )
    internal_features: List[LesionInternalFeature] = Field(
        default_factory=list,
        description=(
            "Internal features described for the lesion (e.g. necrotic, calcified, sclerotic, lytic). "
            "Multi-valued: include all features mentioned, empty if not characterised."
        ),
    )
    margin: LesionMargin = Field(
        description="Edge appearance of the lesion (BI-RADS margin lexicon). Use NOT_DESCRIBED if not characterised."
    )
    shape: LesionShape = Field(
        description="Overall contour shape of the lesion (BI-RADS shape descriptor). Use NOT_DESCRIBED if not characterised."
    )
    density: Optional[str] = Field(
        None,
        description="Density / signal / echogenicity descriptors as reported",
    )

class NonCancerFinding(BaseModel):
    finding_type: NonCancerFindingType = Field(
        description="Mechanism-based finding type. Use OTHER for findings not in the enum."
    )
    anatomical_site: AnatomicalSite = Field(
        description="Anatomical site of the finding"
    )
    anatomical_site_desc: Optional[str] = Field(
        None,
        description="Direct extract of descriptive text for the finding's anatomical site",
    )
    finding_desc: str = Field(
        description="Direct extract of descriptive text for the finding"
    )
    is_cancer_lesion_related: bool = Field(
        False,
        description="True if the finding is caused by or directly related to a cancer lesion in the report (e.g. malignant biliary obstruction, tumour-related lobar collapse, pathological fracture, malignant effusion)",
    )


# FINAL MODEL

class OncoRadModel(BaseModel):
    is_radiology_report: bool = Field(
        description="True only if the document is a radiology report"
    )
    is_oncology_related: bool = Field(
        description="True only if the report concerns a patient with cancer"
    )
    scan_metadata: Optional[ScanMetadata] = Field(
        None,
        description="Scan-level metadata; None if not a radiology report",
    )
    image_rationale: Optional[ImageRationale] = Field(
        None,
        description="Rationale for the imaging study (screening, diagnosis/staging, follow-up, interventional, etc.)",
    )
    indication_desc: Optional[str] = Field(
        None,
        description="Direct extract of descriptive text for the clinical indication / reason for the scan",
    )
    cancer_status: Optional[CancerStatus] = Field(
        None,
        description="Patient-level cancer status summary; None if not oncology related",
    )
    cancer_lesions: Optional[List[CancerLesion]] = Field(
        None, description="All cancer-related lesions described in the report"
    )
    non_cancer_findings: Optional[List[NonCancerFinding]] = Field(
        None, description="Other non-cancer findings described in the report"
    )
    report_summary: Optional[str] = Field(
        None,
        description="Short free-text overall summary of the report; None if not a radiology report",
    )
