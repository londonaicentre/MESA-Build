from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

# TYPES & ENUMS

Year = Annotated[int, Field(ge=1900, le=2100)]


class ScanModality(str, Enum):
    XRAY = "xray"
    CT = "ct"
    OTHER = "other"


class ScanProtocol(str, Enum):
    """Protocol or view used. Multi-valued."""

    OTHER = "other"

    # X-ray views
    PA = "pa"
    AP = "ap"
    LATERAL = "lateral"
    SUPINE = "supine"
    MOBILE = "mobile"  # incl. portable

    # CT protocols
    HRCT = "hrct"  # high-resolution / ILD protocol
    CTPA = "ctpa"  # CT pulmonary angiogram
    CT_AORTOGRAM = "ct_aortogram"
    LOW_DOSE = "low_dose"  # incl. lung cancer screening, targeted lung health check
    EXPIRATORY = "expiratory"
    PRONE = "prone"


class ScanContrast(str, Enum):
    WITHOUT_CONTRAST = "without_contrast"
    WITH_CONTRAST = "with_contrast"


class Assertion(str, Enum):
    """How is the whole entity asserted?
    EDGE CASE: 'effusion has resolved' is positive with change 'resolved'."""

    POSITIVE = "positive"  # stated, incl. 'in keeping with', 'consistent with', 'likely', 'probable'
    NEGATED = (
        "negated"  # explicitly absent (e.g. "no effusion", "absence of honeycombing")
    )
    UNCERTAIN = "uncertain"  # 'possible', 'may represent', '?', 'cannot be excluded', or one of a differential


class ComparativeChange(str, Enum):
    """Explicit mention of change vs previous scan based on radiologist determination"""

    NEW = "new"
    PROGRESSIVE = "progressive"
    STABLE = "stable"
    IMPROVING = "improving"
    RESOLVED = "resolved"
    MIXED = "mixed"
    INDETERMINATE = "indeterminate"  # explicit statement that change cannot be assessed
    NOT_STATED = "not_stated"


class Extent(str, Enum):
    """Severity, extent or size as the radiologist grades it"""

    MILD_SMALL = (
        "mild_small"  # e.g. 'trace', 'minimal', 'subtle', 'mild', 'small', 'minor'
    )
    MODERATE = "moderate"
    LARGE_EXTENSIVE = "large_extensive"  # e.g. 'large', 'extensive', 'severe', 'marked', 'widespread', 'gross'
    NOT_STATED = "not_stated"


class AnatomicalSiteDistribution(str, Enum):
    """Thoracic site or distribution. Multi-valued: every entry that is true goes in,
    broad and specific together."""

    OTHER = "other"

    # Lung (side)
    LEFT_LUNG = "left_lung"
    RIGHT_LUNG = "right_lung"
    LUNG_LOCATION_UNSPECIFIED = "lung_location_unspecified"

    # Lung (lobe)
    RIGHT_UPPER_LOBE = "right_upper_lobe"
    RIGHT_MIDDLE_LOBE = "right_middle_lobe"
    RIGHT_LOWER_LOBE = "right_lower_lobe"
    LEFT_UPPER_LOBE = "left_upper_lobe"
    LINGULA = "lingula"
    LEFT_LOWER_LOBE = "left_lower_lobe"

    # Lung (zone)
    LEFT_UPPER_ZONE = "left_upper_zone"  # incl. apex, apical
    LEFT_MID_ZONE = "left_mid_zone"
    LEFT_LOWER_ZONE = "left_lower_zone"  # incl. base, basal
    RIGHT_UPPER_ZONE = "right_upper_zone"  # incl. apex, apical
    RIGHT_MID_ZONE = "right_mid_zone"
    RIGHT_LOWER_ZONE = "right_lower_zone"  # incl. base, basal

    # Lung (distribution)
    LEFT_SUBPLEURAL_PERIPHERAL = "left_subpleural_peripheral"
    RIGHT_SUBPLEURAL_PERIPHERAL = "right_subpleural_peripheral"
    LEFT_CENTRAL_PERIBRONCHOVASCULAR = "left_central_peribronchovascular"
    RIGHT_CENTRAL_PERIBRONCHOVASCULAR = "right_central_peribronchovascular"
    LEFT_PERIHILAR = "left_perihilar"
    RIGHT_PERIHILAR = "right_perihilar"
    LEFT_DIFFUSE = "left_diffuse"  # incl. widespread, throughout
    RIGHT_DIFFUSE = "right_diffuse"  # incl. widespread, throughout

    # Airways
    AIRWAYS_LOCATION_UNSPECIFIED = "airways_location_unspecified"
    TRACHEA = "trachea"
    LEFT_MAIN_BRONCHUS = "left_main_bronchus"
    RIGHT_MAIN_BRONCHUS = "right_main_bronchus"

    # Pleura
    LEFT_PLEURA = "left_pleura"
    RIGHT_PLEURA = "right_pleura"

    # Mediastinum / hila
    MEDIASTINUM = "mediastinum"
    LEFT_HILUM = "left_hilum"
    RIGHT_HILUM = "right_hilum"

    # Heart / vessels
    HEART = "heart"
    PERICARDIUM = "pericardium"
    PULMONARY_VESSELS = "pulmonary_vessels"
    AORTA = "aorta"

    # Pulmonary arteries (e.g. embolus level)
    MAIN_PULMONARY_ARTERY = (
        "main_pulmonary_artery"  # incl. trunk, saddle, left/right main
    )
    LOBAR_PULMONARY_ARTERY = "lobar_pulmonary_artery"
    SEGMENTAL_PULMONARY_ARTERY = "segmental_pulmonary_artery"  # incl. subsegmental

    # Oesophagus / diaphragm
    OESOPHAGUS = "oesophagus"
    DIAPHRAGM = "diaphragm"
    LEFT_HEMIDIAPHRAGM = "left_hemidiaphragm"
    RIGHT_HEMIDIAPHRAGM = "right_hemidiaphragm"

    # Chest wall / bone / soft tissue
    LEFT_RIBS = "left_ribs"
    RIGHT_RIBS = "right_ribs"
    STERNUM = "sternum"
    THORACIC_SPINE = "thoracic_spine"
    LEFT_AXILLA = "left_axilla"
    RIGHT_AXILLA = "right_axilla"
    THORACIC_SOFT_TISSUE = "thoracic_soft_tissue"  # incl. chest wall, breast
    THYROID = "thyroid"

    # Upper abdomen
    UPPER_ABDOMEN = "upper_abdomen"
    LIVER = "liver"


class FindingType(str, Enum):
    """Radiological findings, named by the terms radiologists write.
    Broad and specific types may both apply to the same text."""

    OTHER = "other"

    # Fibrosis / interstitial
    FIBROSIS = (
        "fibrosis"  # "fibrotic change", "pulmonary fibrosis", "fibrotic lung disease"
    )
    INTERSTITIAL_CHANGE = (
        "interstitial_change"  # "interstitial markings/shadowing", "reticulonodular"
    )
    RETICULATION = "reticulation"  # "reticular change/opacities/pattern"
    HONEYCOMBING = "honeycombing"
    TRACTION_BRONCHIECTASIS = "traction_bronchiectasis"  # incl. bronchiolectasis
    GROUND_GLASS = "ground_glass"  # "ground glass change/opacification/opacity"
    VOLUME_LOSS = "volume_loss"  # parenchymal shrinkage, not lobar collapse
    ARCHITECTURAL_DISTORTION = "architectural_distortion"
    SEPTAL_THICKENING = "septal_thickening"  # incl. interlobular septal thickening
    INTERSTITIAL_LUNG_ABNORMALITY = "interstitial_lung_abnormality"  # ILA
    PLEUROPARENCHYMAL_FIBROELASTOSIS = "pleuroparenchymal_fibroelastosis"  # PPFE
    SCARRING = (
        "scarring"  # incl. parenchymal band, post-inflammatory / post-infective change
    )

    # ILD patterns (guideline categories)
    UIP_PATTERN = "uip_pattern"  # typical / definite UIP
    PROBABLE_UIP_PATTERN = "probable_uip_pattern"
    INDETERMINATE_FOR_UIP_PATTERN = "indeterminate_for_uip_pattern"
    NON_UIP_PATTERN = "non_uip_pattern"  # incl. 'alternative diagnosis' category
    NSIP_PATTERN = "nsip_pattern"  # cellular or fibrotic NSIP
    OP_PATTERN = "op_pattern"  # organising pneumonia
    HP_PATTERN = "hp_pattern"  # typical / compatible with hypersensitivity pneumonitis

    # Airspace / parenchyma
    OPACITY = (
        "opacity"  # "opacity", "opacification", "shadowing", "haziness", "white-out"
    )
    AIRSPACE_SHADOWING = "airspace_shadowing"  # "airspace opacification/change/disease"
    CONSOLIDATION = "consolidation"
    ATELECTASIS = "atelectasis"  # incl. linear, plate, discoid, subsegmental, dependent
    COLLAPSE = "collapse"  # segmental, lobar or whole lung
    INCREASED_LUNG_MARKINGS = (
        "increased_lung_markings"  # "prominent/coarse lung or bronchovascular markings"
    )
    PULMONARY_OEDEMA = "pulmonary_oedema"  # interstitial or alveolar, incl. bat-wing
    PULMONARY_VENOUS_CONGESTION = (
        "pulmonary_venous_congestion"  # "vascular/venous congestion"
    )
    UPPER_LOBE_DIVERSION = "upper_lobe_diversion"  # incl. vascular redistribution
    KERLEY_B_LINES = "kerley_b_lines"  # incl. septal lines on X-ray
    CRAZY_PAVING = "crazy_paving"
    MOSAIC_ATTENUATION = "mosaic_attenuation"  # incl. mosaic perfusion
    AIR_TRAPPING = "air_trapping"
    EMPHYSEMA = "emphysema"  # any subtype, incl. centrilobular, paraseptal
    BULLA = "bulla"  # incl. bullous change
    CYST = "cyst"  # incl. cystic change
    HYPERINFLATION = "hyperinflation"  # incl. hyperexpanded lungs, flattened diaphragms
    PULMONARY_INFARCT = "pulmonary_infarct"

    # Airways
    BRONCHIECTASIS = "bronchiectasis"  # non-traction
    BRONCHIAL_WALL_THICKENING = (
        "bronchial_wall_thickening"  # incl. peribronchial cuffing
    )
    MUCUS_PLUGGING = "mucus_plugging"  # incl. mucoid impaction, secretions
    TREE_IN_BUD = "tree_in_bud"
    AIRWAY_NARROWING = "airway_narrowing"  # incl. stenosis, tracheobronchomalacia

    # Nodules / masses
    NODULE = "nodule"  # any nodule; incl. unlabelled 'lesion' up to 3cm
    GROUND_GLASS_NODULE = "ground_glass_nodule"  # incl. subsolid, non-solid
    PART_SOLID_NODULE = "part_solid_nodule"
    PERIFISSURAL_NODULE = "perifissural_nodule"  # incl. intrapulmonary lymph node
    MICRONODULES = (
        "micronodules"  # "small/tiny nodules", centrilobular, perilymphatic, miliary
    )
    MASS = "mass"  # incl. unlabelled 'lesion' over 3cm
    CAVITY = "cavity"  # incl. cavitation, cavitating lesion
    CALCIFIED_GRANULOMA = "calcified_granuloma"  # incl. calcified nodule

    # Pleura
    PLEURAL_EFFUSION = "pleural_effusion"  # incl. pleural fluid
    COSTOPHRENIC_ANGLE_BLUNTING = "costophrenic_angle_blunting"
    PLEURAL_THICKENING = "pleural_thickening"  # incl. apical capping
    PLEURAL_NODULARITY = (
        "pleural_nodularity"  # incl. nodular pleural thickening, pleural mass
    )
    PLEURAL_PLAQUE = "pleural_plaque"
    PNEUMOTHORAX = "pneumothorax"

    # Mediastinum / hila
    LYMPHADENOPATHY = "lymphadenopathy"  # "enlarged/prominent/bulky nodes"
    HILAR_ENLARGEMENT = "hilar_enlargement"  # "prominent/bulky hila", "hilar fullness"
    MEDIASTINAL_WIDENING = "mediastinal_widening"
    MEDIASTINAL_SHIFT = "mediastinal_shift"  # incl. tracheal deviation
    PNEUMOMEDIASTINUM = "pneumomediastinum"

    # Vascular / cardiac
    PULMONARY_EMBOLISM = "pulmonary_embolism"  # incl. filling defect, embolus
    RIGHT_HEART_STRAIN = (
        "right_heart_strain"  # incl. RV dilatation, raised RV:LV ratio, septal bowing
    )
    PULMONARY_ARTERY_ENLARGEMENT = "pulmonary_artery_enlargement"
    CARDIOMEGALY = "cardiomegaly"  # incl. enlarged heart / cardiac silhouette
    PERICARDIAL_EFFUSION = "pericardial_effusion"
    UNFOLDED_AORTA = "unfolded_aorta"
    AORTIC_ANEURYSM_DILATATION = "aortic_aneurysm_dilatation"  # incl. ectasia
    AORTIC_DISSECTION = "aortic_dissection"
    ATHEROSCLEROSIS = "atherosclerosis"  # incl. atheroma, vascular calcification
    CORONARY_ARTERY_CALCIFICATION = "coronary_artery_calcification"

    # Oesophagus / diaphragm
    OESOPHAGEAL_DILATATION = "oesophageal_dilatation"  # incl. patulous oesophagus
    HIATUS_HERNIA = "hiatus_hernia"
    DIAPHRAGM_ELEVATION = "diaphragm_elevation"  # incl. eventration

    # Chest wall / bone / soft tissue
    FRACTURE = "fracture"
    BONE_LESION = "bone_lesion"  # lytic or sclerotic
    DEGENERATIVE_CHANGE = "degenerative_change"  # incl. spondylosis
    SUBCUTANEOUS_EMPHYSEMA = "subcutaneous_emphysema"  # incl. surgical emphysema
    THYROID_NODULE_GOITRE = "thyroid_nodule_goitre"

    # Upper abdomen
    PNEUMOPERITONEUM = "pneumoperitoneum"  # incl. free gas under the diaphragm
    HEPATIC_STEATOSIS = "hepatic_steatosis"

    # Treatment / devices
    POST_SURGICAL_CHANGE = (
        "post_surgical_change"  # e.g. sternotomy wires, lobectomy, clips
    )
    LINE_TUBE_DEVICE = "line_tube_device"  # any or unspecified, e.g. "lines and tubes"
    ENDOTRACHEAL_TUBE = "endotracheal_tube"  # ETT
    TRACHEOSTOMY = "tracheostomy"
    NASOGASTRIC_TUBE = "nasogastric_tube"  # incl. NG, NJ, fine-bore feeding tube
    CENTRAL_VENOUS_CATHETER = (
        "central_venous_catheter"  # incl. CVC, IJ/subclavian line, Hickman, portacath
    )
    PICC_LINE = "picc_line"
    CHEST_DRAIN = (
        "chest_drain"  # incl. intercostal drain, pigtail, indwelling pleural catheter
    )
    CARDIAC_DEVICE = "cardiac_device"  # incl. pacemaker, ICD, CRT, loop recorder
    PROSTHETIC_HEART_VALVE = "prosthetic_heart_valve"  # incl. TAVI
    STENT = "stent"  # airway, oesophageal or vascular
    LINE_TUBE_MALPOSITION = (
        "line_tube_malposition"  # incl. "tip in bronchus", "suboptimal position"
    )


class DiagnosisType(str, Enum):
    """Diagnoses offered as the radiologist's interpretation of this scan.
    Directly seen findings (e.g. pulmonary embolism, pneumothorax) are findings only."""

    OTHER = "other"

    # Interstitial lung disease
    INTERSTITIAL_LUNG_DISEASE = "interstitial_lung_disease"  # any or unspecified ILD, incl. "fibrotic lung disease"
    PROGRESSIVE_PULMONARY_FIBROSIS = (
        "progressive_pulmonary_fibrosis"  # named as PPF / PF-ILD
    )
    IDIOPATHIC_PULMONARY_FIBROSIS = "idiopathic_pulmonary_fibrosis"
    CTD_ILD = "ctd_ild"  # connective tissue disease ILD, e.g. scleroderma, RA, myositis
    HYPERSENSITIVITY_PNEUMONITIS = "hypersensitivity_pneumonitis"  # incl. fibrotic HP
    SARCOIDOSIS = "sarcoidosis"
    DRUG_INDUCED_PNEUMONITIS = (
        "drug_induced_pneumonitis"  # incl. immunotherapy pneumonitis
    )
    RADIATION_PNEUMONITIS_FIBROSIS = "radiation_pneumonitis_fibrosis"
    SMOKING_RELATED_ILD = "smoking_related_ild"  # incl. RB-ILD, DIP
    PNEUMOCONIOSIS = "pneumoconiosis"  # incl. asbestosis, silicosis
    ASBESTOSIS = "asbestosis"
    CYSTIC_LUNG_DISEASE = "cystic_lung_disease"  # incl. LAM, LCH, LIP

    # Infection
    PNEUMONIA_INFECTION = "pneumonia_infection"  # incl. LRTI, "infective change"
    COVID_19 = "covid_19"
    ASPIRATION = "aspiration"
    TUBERCULOSIS = "tuberculosis"  # active or previous
    NTM = "ntm"  # non-tuberculous mycobacterial infection
    FUNGAL_INFECTION = "fungal_infection"  # incl. aspergilloma, ABPA, PCP
    EMPYEMA = "empyema"  # incl. complicated parapneumonic effusion
    LUNG_ABSCESS = "lung_abscess"

    # Neoplastic
    MALIGNANCY = "malignancy"  # any or unspecified, incl. "suspicious for neoplasm"
    LUNG_CANCER = "lung_cancer"
    METASTATIC_DISEASE = "metastatic_disease"
    LYMPHANGITIS_CARCINOMATOSA = "lymphangitis_carcinomatosa"
    MESOTHELIOMA = "mesothelioma"
    LYMPHOMA = "lymphoma"

    # Cardiovascular
    HEART_FAILURE_FLUID_OVERLOAD = "heart_failure_fluid_overload"
    PULMONARY_HYPERTENSION = "pulmonary_hypertension"
    CHRONIC_THROMBOEMBOLIC_DISEASE = "chronic_thromboembolic_disease"  # incl. CTEPH

    # Airways
    COPD = "copd"
    SMALL_AIRWAYS_DISEASE = (
        "small_airways_disease"  # incl. bronchiolitis, obliterative bronchiolitis
    )
    CYSTIC_FIBROSIS = "cystic_fibrosis"

    # Other
    ASBESTOS_RELATED_PLEURAL_DISEASE = "asbestos_related_pleural_disease"
    PULMONARY_HAEMORRHAGE = "pulmonary_haemorrhage"
    ARDS = "ards"


# BLOCKS


class ScanMetadata(BaseModel):
    modalities: list[ScanModality] = Field(
        description="Scan modalities used; more than one if the report covers several scans"
    )
    protocols: list[ScanProtocol] = Field(
        default_factory=list,
        description="Scan protocols or views stated in the report; empty if none stated",
    )
    contrast: ScanContrast | None = Field(
        None,
        description="Contrast usage for the scan; None if the report does not say whether contrast was given",
    )
    comparison_year: Year | None = Field(
        None,
        description="Year of the prior scan this scan is compared against, only if explicitly stated",
    )
    image_quality_desc: str | None = Field(
        None,
        description="Direct extract describing image quality or artefact affecting interpretation",
    )


class Finding(BaseModel):
    finding: FindingType = Field(
        description="Finding type, named by the report's wording. Use 'other' for findings not in the enum."
    )
    finding_desc: str = Field(
        description="Direct extract of descriptive text for the finding"
    )
    assertion: Assertion = Field(
        description="Assertion of the whole finding, including any change: positive if stated, 'in keeping with', 'likely' or 'probable' (e.g. 'effusion has resolved' is positive with change 'resolved'); uncertain if 'possible', 'may represent', queried, 'cannot be excluded' or in a differential; negated if explicitly absent"
    )
    change: ComparativeChange = Field(
        ComparativeChange.NOT_STATED,
        description="Change of this finding compared to prior scan",
    )
    change_desc: str | None = Field(
        None, description="Direct extract of descriptive text for the change"
    )
    extent: Extent = Field(
        Extent.NOT_STATED,
        description="Severity, extent or size as graded by the radiologist; for a range (e.g. 'mild to moderate') use the higher grade",
    )
    extent_desc: str | None = Field(
        None,
        description="Direct extract describing the finding's severity, extent, or size (e.g. 'exuberant', 'extensive', '8mm')",
    )
    anatomical_site_distribution: list[AnatomicalSiteDistribution] = Field(
        default_factory=list,
        description="Every site and distribution true of the finding, broad and specific together; a lung lobe, zone or distribution also gets its lung side (e.g. 'right basal subpleural reticulation' is right_lung, right_lower_zone, right_subpleural_peripheral). Bilateral descriptions (e.g. 'bibasal') get both left and right entries. Empty if not stated",
    )
    anatomical_site_distribution_desc: str | None = Field(
        None,
        description="Direct extract naming the finding's anatomical site or distribution",
    )


class Diagnosis(BaseModel):
    diagnosis: DiagnosisType = Field(
        description="Diagnosis type. Use 'other' for diagnoses not in the enum."
    )
    diagnosis_desc: str = Field(
        description="Direct extract of descriptive text for the diagnosis"
    )
    assertion: Assertion = Field(
        description="Positive if stated, 'in keeping with', 'consistent with', 'likely' or 'probable'; uncertain if 'possible', 'may represent', queried, 'cannot be excluded' or in a differential; negated if excluded"
    )


class Score(BaseModel):
    score_name_desc: str = Field(
        description="Direct extract naming the scoring system or classification (e.g. 'Lung-RADS', 'Brock', 'CO-RADS')"
    )
    score_value_desc: str = Field(
        description="Direct extract of the score value or category assigned"
    )


# FINAL MODEL


class ChestRadModel(BaseModel):
    is_radiology_report: bool = Field(
        description="True only if the document is a radiology report"
    )
    is_thoracic_imaging: bool = Field(
        description="True only if the scan images the chest (e.g. chest X-ray, CT chest, HRCT)"
    )
    is_unremarkable_study: bool = Field(
        description="True if the radiologist reports the study as normal or without significant abnormality (e.g. 'normal chest', 'lungs clear', 'no acute cardiopulmonary abnormality' with no other findings)"
    )
    scan_metadata: ScanMetadata | None = Field(
        None,
        description="Scan-level metadata; None if not a radiology report",
    )
    overall_change: ComparativeChange = Field(
        ComparativeChange.NOT_STATED,
        description="Radiologist's gestalt change of the whole study vs prior scan",
    )
    overall_change_desc: str | None = Field(
        None, description="Direct extract of descriptive text for the overall change"
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Thoracic findings described in the report, including explicitly negated findings. Text matching a broad and a specific type (e.g. 'fibrotic reticulation') gets an entry for each",
    )
    diagnoses: list[Diagnosis] = Field(
        default_factory=list,
        description="Diagnoses the radiologist offers from this scan; exclude diagnoses stated only in the clinical history",
    )
    scores: list[Score] = Field(
        default_factory=list,
        description="Scoring systems or classifications the radiologist assigns to this scan",
    )
