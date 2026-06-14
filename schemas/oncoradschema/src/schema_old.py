from enum import Enum
from typing import List, Optional, Annotated
from pydantic import BaseModel, Field


# Enums

Year = Annotated[int, Field(ge=1900, le=2100)]

class Modality(str, Enum):
    CT = "CT"
    MRI = "MRI"
    PET_CT = "PET-CT"
    PET_MRI = "PET-MRI"
    SPECT = "SPECT"
    SPECT_CT = "SPECT-CT"
    US = "Ultrasound"
    XR = "X-Ray"
    FLUOROSCOPY = "Fluoroscopy"
    MAMMOGRAPHY = "Mammography"
    DEXA = "DEXA"
    INTERVENTIONAL = "Interventional Radiology"
    OTHER = "Other"

class Contrast(str, Enum):
    WITH = "with_contrast"
    WITHOUT = "without_contrast"
    DUAL_PHASE = "dual_phase"
    TRIPLE_PHASE = "triple_phase"
    NOT_APPLICABLE = "not_applicable"

class Region(str, Enum):
    HEAD = "Head"
    NECK = "Neck"            
    CHEST = "Chest"                 
    ABDOMEN = "Abdomen"
    PELVIS = "Pelvis"
    SPINE = "Spine"
    UPPER_LIMB = "Upper Limb"
    LOWER_LIMB = "Lower Limb"
    WHOLE_BODY = "Whole Body"
    CARDIAC = "Cardiac"
    BREAST = "Breast"
    VASC = "Vessel"
    OTHER = "Other"

class Laterality(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"
    BILATERAL = "Bilateral"
    MIDLINE = "Midline"
    NOT_APPLICABLE = "N/A"






class ComparativeChange(str, Enum):
    NEW = "New (no prior)"
    PROGRESSIVE = "Progressive"
    STABLE = "Stable"
    IMPROVING = "Improving"
    RESOLVED = "Resolved"
    MIXED = "Mixed response"
    INDETERMINATE = "Indeterminate"

class RECISTResponse(str, Enum):
    """
    RECIST 1.1 treatment response categories on oncological follow-up imaging.
    """
    COMPLETE_RESPONSE = "Complete Response (CR)"      # disappearance of all target lesions
    PARTIAL_RESPONSE = "Partial Response (PR)"        # ≥30% decrease in sum of diameters
    STABLE_DISEASE = "Stable Disease (SD)"            # neither PR nor PD criteria met
    PROGRESSIVE_DISEASE = "Progressive Disease (PD)"  # ≥20% increase or new lesions
    NOT_EVALUABLE = "Not Evaluable (NE)"              # cannot assess — missing, artefact, wrong modality
    NOT_APPLICABLE = "Not Applicable"                 # not a target lesion / not a follow-up scan



class Topography(str, Enum):
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




                
class LesionMargin(str, Enum):
    WELL_DEFINED = "Well-defined"          
    CIRCUMSCRIBED = "Circumscribed"       
    ILL_DEFINED = "Ill-defined"       
    INDISTINCT = "Indistinct"             
    OBSCURED = "Obscured"                
    UNKNOWN = "Unknown"

class LesionCharacter(str, Enum):
    SMOOTH = "Smooth"                      
    LOBULATED = "Lobulated"               
    IRREGULAR = "Irregular"            
    ANGULAR = "Angular"                 
    SPICULATED = "Spiculated"            
    INFILTRATIVE = "Infiltrative"                         
    ENCAPSULATED = "Encapsulated"        
    UNKNOWN = "Unknown"

class LesionMorphology(str, Enum):
    SOLID = "Solid"
    CYSTIC = "Cystic"
    SOLID_CYSTIC = "Mixed solid-cystic"
    GROUND_GLASS = "Ground glass"
    PART_SOLID = "Part-solid"
    CALCIFIED = "Calcified"
    NECROTIC = "Necrotic"
    HAEMORRHAGIC = "Haemorrhagic"
    MUCINOUS = "Mucinous"

class Density(str, Enum):
    # CT — Hounsfield-based
    HYPERDENSE = "Hyperdense"
    ISODENSE = "Isodense"
    HYPODENSE = "Hypodense"
    FAT_DENSITY = "Fat density"
    CALCIFIC = "Calcific density"
    # MRI — signal-based
    T1_HYPERINTENSE = "T1 hyperintense"
    T1_HYPOINTENSE = "T1 hypointense"
    T2_HYPERINTENSE = "T2 hyperintense"
    T2_HYPOINTENSE = "T2 hypointense"
    RESTRICTED_DIFFUSION = "Restricted diffusion"
    # US — echo-based
    HYPERECHOIC = "Hyperechoic"
    HYPOECHOIC = "Hypoechoic"
    ISOECHOIC = "Isoechoic"
    ANECHOIC = "Anechoic"
    # Generic
    HETEROGENEOUS = "Heterogeneous"
    HOMOGENEOUS = "Homogeneous"


class RecommendationType(str, Enum):
    FOLLOW_UP_IMAGING = "Follow-up imaging"
    MDT_DISCUSSION = "MDT discussion"
    BIOPSY = "Biopsy / tissue sampling"
    CLINICAL_CORRELATION = "Clinical correlation"
    URGENT_REFERRAL = "Urgent referral"
    NO_FURTHER_ACTION = "No further action"
    ADDITIONAL_IMAGING = "Additional / complementary imaging"
    INTERVENTION = "Radiological intervention"
    SURVEILLANCE = "Surveillance programme"
    OTHER = "Other"






class Metadata(BaseModel):
    modality: Modality
    contrast: Optional[Contrast] = None
    region: Optional[List[Region]] = Field(
        None, description="regions captured by scan"
    )
    

class PrevStudy(BaseModel):
    year_prior: Optional[Year] = Field(
        None, description="Year of previous imaging"
    )
    modality: Optional[Modality] = None
    overall_change: Optional[ComparativeChange] = Field(
        None, description="Change compared to previous study"
    )
    raw_text: Optional[str] = Field(
        None, description="Verbatim comparison sentence(s) from report"
    )

class LesionSize(BaseModel):

    longest_diameter_mm: Optional[float] = Field(
        None, 
        description="Longest diameter of lesion"
    )
    perpendicular_diameter_mm: Optional[float] = Field(
        None, 
        description="Second dimension if reported as '34 x 28mm'"
    )
    third_dimension_mm: Optional[float] = Field(
        None, 
        description="Third dimension if reported"
    )
    
    volume_ml: Optional[float] = Field(
        None, 
        description="Volume in millilitres (mL) or cubic centimetres (cc) if explicitly reported"
    )

class DiseaseSpecificScore(BaseModel):
    scoring_system: str = Field(
        ..., description="Name of scoring / staging system e.g. 'TNM', 'PI-RADS v2.1', 'Bosniak 2019'"
    )
    score_or_stage: str = Field(
        ..., description="The score or stage assigned e.g. 'T3bN1M0', 'PI-RADS 4', 'Category III'"
    )
    raw_text: str = Field(
        ..., description="Verbatim staging / scoring snippet from report"
)
    

# Lesion Block
class Lesion(BaseModel):
    is_largest: bool = Field(
        False, description="True if largest lesion on the report"
    )

    location: Optional[str] = Field(
        None, description="Reported location of lesion"
    )

    size: Optional[LesionSize] = Field(
        None, description="Size of lesion"
    )

    margin: Optional[LesionMargin] = Field(
        None, description="Edge appearance of the lesion"
    )

    character: Optional[LesionCharacter] = Field(
        None, description="Character of the lesion"
    )

    morphology: Optional[LesionMorphology] = Field(
        None, description="Morphology of the lesion"
    )

    tissue_density: Optional[Density] = Field(
        None, description="Density of tissue dependent on imaging type"
    )

    change: Optional[ComparativeChange] = Field(
        None, description="Change of lesion compared to previous study"
    )

    is_node: bool = Field(
        False, description="True only if lesion is a lymph node"
    )

    is_metastasis: bool = Field(
        False, description="True only if lesion is a metastasis from a tumour"
    )

  
class BaseFinding(BaseModel):
    topography: Topography = Field(
        description="Most suitable anatomical site of primary finding. Use OTHER where there is ambiguity, or no suitable found in enum."
    )
    laterality: Optional[Laterality] = None

    number_of_lesions: Optional[int] = Field(
        None,
        description="Number of discrete lesions; None if not reported"
    )

    lesion_distribution: Optional[str] = Field(
    None,
    description="Distribution: focal, multifocal, diffuse, scattered, etc."
    )

    lesion: Optional[List[Lesion]] = Field(
        None, description="Characterise each lesion mentioned on the report"
    )

    vascular_changes: Optional[str] = Field(
    None,
    description="Vasculature findings from report e.g. 'tumour thrombus', 'arterial stenosis'"
    )

    imaging_signs: Optional[List[str]] = Field(
        None,
        description=(
            "Specific imaging signs or features cited e.g. 'calcification', 'emphysematous change', 'hyperinflation', 'flattened diaphragm' "
        )
    )


# Tumour Finding Block
class TumourFinding(BaseFinding):
    summary: str = Field(
        ..., description="Verbatim sentences summarising finding from report"
    )
    
    disease_score: Optional[List[DiseaseSpecificScore]] = Field(
        None,
        description=(
            "Any structured scoring mentioned in the report e.g. TNM, BI-RADS "
        )
    )
    recist_response: Optional[RECISTResponse] = Field(
        None,
        description="RECIST 1.1 response category if reported"
    )

# Non-tumour Finding
class NonTumourFinding(BaseFinding):
    summary: Optional[str] = Field(
        None, description="Verbatim sentences summarising finding from report"
    )

# Final Model
class RadReport(BaseModel):
    metadata: Metadata = Field(
        ..., description="Examination type, modality, region, contrast"
    )
    prior_imaging: Optional[List[PrevStudy]] = Field(
        None,
        description=(
            "Comparison with prior studies — None if no prior imaging referenced"
        )
    )
    tumour_findings: Optional[List[TumourFinding]] = Field(
        None,
        description=(
            "All tumour related findings from report"
        )
    )
    non_tumour_findings: Optional[List[NonTumourFinding]] = Field(
        None,
        description=(
            "All other non-tumour related findings from report" 
        )
    )
    recommendations: Optional[List[RecommendationType]] = Field(
        None,
        description="Structured list of recommendations from the report impression"
    )


    impression_summary: Optional[str] = Field(
        None,
        description=(
            "Free-text overall impression verbatim from report"
        )
    )

    report_normal: bool = Field(
        False,
        description="True only if the report is explicitly normal with no significant findings"
    )