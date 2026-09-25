from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

# TYPES

Year = Annotated[int, Field(ge=1900, le=2100)]
Month = Annotated[int, Field(ge=1, le=12)]


# ENUMS - PROCEDURE


class ProcedureType(str, Enum):
    """Operation performed, named by what was done and where, not by the pathology
    treated. Grouped by body system, loosely following OPCS-4 chapter order.
    One entry per distinct procedure performed. The access route (e.g. laparotomy,
    laparoscopy, craniotomy, arthroscopy, endoscopy) is recorded in approach, not
    as a procedure. Scope values (upper_gi_endoscopy, lower_gi_endoscopy,
    bronchoscopy, laryngoscopy, cystoscopy, ureteroscopy, hysteroscopy) are coded
    only when no other listed procedure was performed through the scope."""

    OTHER = "other"

    # Nervous system
    DECOMPRESSIVE_CRANIECTOMY = "decompressive_craniectomy"
    EXCISION_OF_BRAIN_LESION = "excision_of_brain_lesion"
    BRAIN_BIOPSY = "brain_biopsy"  # stereotactic or open
    EVACUATION_OF_INTRACRANIAL_HAEMATOMA = (
        "evacuation_of_intracranial_haematoma"  # extradural, subdural or intracerebral
    )
    REPAIR_OF_CEREBRAL_ANEURYSM = (
        "repair_of_cerebral_aneurysm"  # clipping, coiling or flow diversion
    )
    VENTRICULAR_SHUNT_PROCEDURE = (
        "ventricular_shunt_procedure"  # insertion, revision or removal
    )
    EXTERNAL_VENTRICULAR_DRAIN_INSERTION = "external_ventricular_drain_insertion"
    INSERTION_OF_INTRACRANIAL_PRESSURE_MONITOR = (
        "insertion_of_intracranial_pressure_monitor"
    )
    ENDOSCOPIC_THIRD_VENTRICULOSTOMY = "endoscopic_third_ventriculostomy"
    NEUROMODULATION_DEVICE_INSERTION = "neuromodulation_device_insertion"  # e.g. deep brain, vagal nerve or spinal cord stimulator, intrathecal pump
    MICROVASCULAR_DECOMPRESSION_OF_CRANIAL_NERVE = (
        "microvascular_decompression_of_cranial_nerve"
    )
    REPAIR_OF_DURA = "repair_of_dura"
    PERIPHERAL_NERVE_REPAIR_OR_GRAFT = "peripheral_nerve_repair_or_graft"
    PERIPHERAL_NERVE_DECOMPRESSION = (
        "peripheral_nerve_decompression"  # e.g. carpal or cubital tunnel
    )
    SYMPATHECTOMY = "sympathectomy"

    # Endocrine and breast
    TOTAL_THYROIDECTOMY = "total_thyroidectomy"
    SUBTOTAL_THYROIDECTOMY = "subtotal_thyroidectomy"
    THYROID_LOBECTOMY = "thyroid_lobectomy"  # hemithyroidectomy
    COMPLETION_THYROIDECTOMY = "completion_thyroidectomy"
    PARATHYROIDECTOMY = "parathyroidectomy"
    ADRENALECTOMY = "adrenalectomy"
    MASTECTOMY = "mastectomy"  # simple, skin sparing or nipple sparing
    WIDE_LOCAL_EXCISION_OF_BREAST = "wide_local_excision_of_breast"
    EXCISION_OF_BREAST_LESION = "excision_of_breast_lesion"
    BREAST_RECONSTRUCTION = "breast_reconstruction"  # implant or flap based
    BREAST_AUGMENTATION = "breast_augmentation"
    BREAST_REDUCTION = "breast_reduction"
    MICRODOCHECTOMY = "microdochectomy"

    # Eye
    CATARACT_EXTRACTION = (
        "cataract_extraction"  # incl. lens implant at the same sitting
    )
    SECONDARY_INTRAOCULAR_LENS_INSERTION = "secondary_intraocular_lens_insertion"
    VITREORETINAL_SURGERY = (
        "vitreoretinal_surgery"  # e.g. vitrectomy, scleral buckle, retinopexy
    )
    OPHTHALMIC_LASER_PROCEDURE = (
        "ophthalmic_laser_procedure"  # e.g. YAG capsulotomy, retinal photocoagulation
    )
    TRABECULECTOMY = "trabeculectomy"
    INSERTION_OF_GLAUCOMA_DRAINAGE_DEVICE = "insertion_of_glaucoma_drainage_device"
    CORNEAL_GRAFT = "corneal_graft"  # penetrating or lamellar
    STRABISMUS_SURGERY = "strabismus_surgery"
    EYELID_SURGERY = (
        "eyelid_surgery"  # e.g. ptosis, entropion, ectropion, blepharoplasty
    )
    ENUCLEATION_OR_EVISCERATION_OF_EYE = "enucleation_or_evisceration_of_eye"

    # Ear, nose and throat
    MYRINGOTOMY_WITH_GROMMET_INSERTION = "myringotomy_with_grommet_insertion"
    TYMPANOPLASTY = "tympanoplasty"  # incl. myringoplasty
    MASTOIDECTOMY = "mastoidectomy"
    STAPEDECTOMY = "stapedectomy"
    COCHLEAR_IMPLANT_INSERTION = "cochlear_implant_insertion"
    BONE_ANCHORED_HEARING_AID_INSERTION = "bone_anchored_hearing_aid_insertion"
    PINNAPLASTY = "pinnaplasty"
    SEPTOPLASTY = "septoplasty"
    FUNCTIONAL_ENDOSCOPIC_SINUS_SURGERY = "functional_endoscopic_sinus_surgery"
    TONSILLECTOMY = "tonsillectomy"
    ADENOIDECTOMY = "adenoidectomy"
    LARYNGOSCOPY = "laryngoscopy"  # diagnostic or operative, incl. microlaryngoscopy
    LARYNGECTOMY = "laryngectomy"
    TRACHEOSTOMY = "tracheostomy"  # surgical or percutaneous
    PAROTIDECTOMY = "parotidectomy"
    EXCISION_OF_SUBMANDIBULAR_GLAND = "excision_of_submandibular_gland"
    NECK_DISSECTION = "neck_dissection"

    # Mouth
    DENTAL_EXTRACTION = "dental_extraction"
    EXCISION_OF_ORAL_LESION = "excision_of_oral_lesion"
    GLOSSECTOMY = "glossectomy"
    FRENULOPLASTY = "frenuloplasty"

    # Thorax
    LOBECTOMY_OF_LUNG = "lobectomy_of_lung"
    PNEUMONECTOMY = "pneumonectomy"
    SEGMENTECTOMY_OF_LUNG = "segmentectomy_of_lung"
    WEDGE_RESECTION_OF_LUNG = "wedge_resection_of_lung"
    PLEURODESIS = "pleurodesis"
    PLEURECTOMY = "pleurectomy"
    DECORTICATION_OF_LUNG = "decortication_of_lung"
    INSERTION_OF_CHEST_DRAIN = "insertion_of_chest_drain"
    BRONCHOSCOPY = "bronchoscopy"  # diagnostic or therapeutic, incl. EBUS
    MEDIASTINOSCOPY = "mediastinoscopy"

    # Upper digestive system
    OESOPHAGECTOMY = "oesophagectomy"  # incl. oesophagogastrectomy
    OESOPHAGEAL_MYOTOMY = "oesophageal_myotomy"  # e.g. Heller, POEM
    FUNDOPLICATION = "fundoplication"
    HIATUS_HERNIA_REPAIR = "hiatus_hernia_repair"
    REPAIR_OF_STOMACH_OR_DUODENUM = (
        "repair_of_stomach_or_duodenum"  # e.g. oversew or patch repair of an ulcer
    )
    TOTAL_GASTRECTOMY = "total_gastrectomy"
    PARTIAL_GASTRECTOMY = "partial_gastrectomy"  # distal, subtotal or wedge
    GASTROJEJUNOSTOMY = "gastrojejunostomy"
    PYLOROPLASTY_OR_PYLOROMYOTOMY = "pyloroplasty_or_pyloromyotomy"
    BARIATRIC_SURGERY = "bariatric_surgery"  # e.g. sleeve gastrectomy, gastric bypass, band insertion or removal
    GASTROSTOMY_INSERTION = "gastrostomy_insertion"  # PEG, RIG or surgical
    UPPER_GI_ENDOSCOPY = "upper_gi_endoscopy"  # diagnostic or therapeutic

    # Lower digestive system
    RIGHT_HEMICOLECTOMY = "right_hemicolectomy"  # incl. extended
    TRANSVERSE_COLECTOMY = "transverse_colectomy"
    LEFT_HEMICOLECTOMY = "left_hemicolectomy"
    SIGMOID_COLECTOMY = "sigmoid_colectomy"
    SUBTOTAL_OR_TOTAL_COLECTOMY = "subtotal_or_total_colectomy"
    PANPROCTOCOLECTOMY = "panproctocolectomy"
    HARTMANNS_PROCEDURE = "hartmanns_procedure"
    REVERSAL_OF_HARTMANNS_PROCEDURE = "reversal_of_hartmanns_procedure"
    ANTERIOR_RESECTION_OF_RECTUM = "anterior_resection_of_rectum"
    ABDOMINOPERINEAL_RESECTION_OF_RECTUM = "abdominoperineal_resection_of_rectum"
    TRANSANAL_EXCISION_OF_RECTAL_LESION = "transanal_excision_of_rectal_lesion"
    ILEOCAECAL_RESECTION = "ileocaecal_resection"
    SMALL_BOWEL_RESECTION = "small_bowel_resection"
    STRICTUROPLASTY = "stricturoplasty"
    FORMATION_OF_ILEOSTOMY = "formation_of_ileostomy"  # loop or end
    FORMATION_OF_COLOSTOMY = "formation_of_colostomy"  # loop or end
    CLOSURE_OF_STOMA = "closure_of_stoma"
    ILEOANAL_POUCH_FORMATION = "ileoanal_pouch_formation"
    APPENDICECTOMY = "appendicectomy"
    ADHESIOLYSIS = "adhesiolysis"
    LOWER_GI_ENDOSCOPY = (
        "lower_gi_endoscopy"  # colonoscopy or sigmoidoscopy, diagnostic or therapeutic
    )
    HAEMORRHOID_PROCEDURE = "haemorrhoid_procedure"  # e.g. haemorrhoidectomy, stapled haemorrhoidopexy, banding
    LATERAL_SPHINCTEROTOMY = "lateral_sphincterotomy"
    FISTULOTOMY = "fistulotomy"
    INSERTION_OF_SETON = "insertion_of_seton"
    INGUINAL_HERNIA_REPAIR = "inguinal_hernia_repair"
    FEMORAL_HERNIA_REPAIR = "femoral_hernia_repair"
    VENTRAL_HERNIA_REPAIR = "ventral_hernia_repair"  # any abdominal wall hernia, e.g. umbilical, epigastric, incisional, parastomal

    # Hepatobiliary, pancreas and spleen
    CHOLECYSTECTOMY = "cholecystectomy"  # incl. subtotal
    CHOLECYSTOSTOMY = "cholecystostomy"
    BILE_DUCT_EXPLORATION = "bile_duct_exploration"
    HEPATICOJEJUNOSTOMY = "hepaticojejunostomy"
    ERCP = "ercp"  # diagnostic or therapeutic
    PERCUTANEOUS_TRANSHEPATIC_BILIARY_PROCEDURE = (
        "percutaneous_transhepatic_biliary_procedure"  # PTC, biliary drain or stent
    )
    MAJOR_HEPATECTOMY = "major_hepatectomy"  # e.g. right or left hepatectomy
    MINOR_LIVER_RESECTION = "minor_liver_resection"  # e.g. segmentectomy, wedge
    LIVER_TRANSPLANTATION = "liver_transplantation"
    PANCREATICODUODENECTOMY = "pancreaticoduodenectomy"
    DISTAL_PANCREATECTOMY = "distal_pancreatectomy"
    TOTAL_PANCREATECTOMY = "total_pancreatectomy"
    PANCREATIC_NECROSECTOMY = "pancreatic_necrosectomy"
    SPLENECTOMY = "splenectomy"  # total or partial

    # Heart
    CORONARY_ARTERY_BYPASS_GRAFT = "coronary_artery_bypass_graft"
    CORONARY_ANGIOGRAPHY = "coronary_angiography"
    PERCUTANEOUS_CORONARY_INTERVENTION = "percutaneous_coronary_intervention"
    AORTIC_VALVE_REPLACEMENT = "aortic_valve_replacement"  # surgical or transcatheter
    MITRAL_VALVE_REPAIR_OR_REPLACEMENT = "mitral_valve_repair_or_replacement"
    TRICUSPID_OR_PULMONARY_VALVE_REPAIR_OR_REPLACEMENT = (
        "tricuspid_or_pulmonary_valve_repair_or_replacement"
    )
    CARDIAC_DEVICE_PROCEDURE = "cardiac_device_procedure"  # pacemaker, ICD or CRT insertion, generator change or lead extraction
    CARDIAC_ABLATION_OR_ELECTROPHYSIOLOGY_STUDY = (
        "cardiac_ablation_or_electrophysiology_study"  # catheter or surgical
    )
    REPAIR_OF_SEPTAL_DEFECT = "repair_of_septal_defect"  # surgical or device closure
    PERICARDIAL_PROCEDURE = (
        "pericardial_procedure"  # pericardiocentesis, window or pericardiectomy
    )
    CARDIAC_TRANSPLANTATION = "cardiac_transplantation"
    MECHANICAL_CIRCULATORY_SUPPORT_INSERTION = (
        "mechanical_circulatory_support_insertion"  # e.g. VAD, IABP, ECMO cannulation
    )

    # Arteries and veins
    AORTIC_REPAIR = "aortic_repair"  # open or endovascular, any segment
    CAROTID_ENDARTERECTOMY = "carotid_endarterectomy"
    ARTERIAL_BYPASS = "arterial_bypass"
    ANGIOPLASTY_OR_STENTING = "angioplasty_or_stenting"  # non-coronary
    EMBOLECTOMY_OR_THROMBECTOMY = "embolectomy_or_thrombectomy"  # arterial or venous
    ARTERIOVENOUS_ACCESS_FORMATION = (
        "arteriovenous_access_formation"  # fistula or graft
    )
    VARICOSE_VEIN_PROCEDURE = "varicose_vein_procedure"  # any technique
    INSERTION_OF_INFERIOR_VENA_CAVA_FILTER = "insertion_of_inferior_vena_cava_filter"
    VASCULAR_ACCESS_DEVICE_INSERTION = (
        "vascular_access_device_insertion"  # CVC, PICC, tunnelled line, port
    )
    VASCULAR_ACCESS_DEVICE_REMOVAL = "vascular_access_device_removal"

    # Urinary
    RADICAL_NEPHRECTOMY = "radical_nephrectomy"
    PARTIAL_NEPHRECTOMY = "partial_nephrectomy"
    SIMPLE_NEPHRECTOMY = "simple_nephrectomy"
    NEPHROURETERECTOMY = "nephroureterectomy"
    RENAL_TRANSPLANTATION = "renal_transplantation"
    PYELOPLASTY = "pyeloplasty"
    NEPHROSTOMY_INSERTION = "nephrostomy_insertion"
    PERCUTANEOUS_NEPHROLITHOTOMY = "percutaneous_nephrolithotomy"
    URETEROSCOPY = "ureteroscopy"  # diagnostic or therapeutic
    URETERIC_STENT_PROCEDURE = (
        "ureteric_stent_procedure"  # insertion, exchange or removal
    )
    URETERIC_REIMPLANTATION = "ureteric_reimplantation"
    RADICAL_CYSTECTOMY = "radical_cystectomy"
    PARTIAL_CYSTECTOMY = "partial_cystectomy"
    URINARY_DIVERSION = "urinary_diversion"  # e.g. ileal conduit, neobladder
    TRANSURETHRAL_RESECTION_OF_BLADDER_TUMOUR = (
        "transurethral_resection_of_bladder_tumour"
    )
    CYSTOSCOPY = "cystoscopy"  # diagnostic or therapeutic, flexible or rigid
    INSERTION_OF_SUPRAPUBIC_CATHETER = "insertion_of_suprapubic_catheter"
    TRANSURETHRAL_PROSTATE_PROCEDURE = "transurethral_prostate_procedure"  # e.g. TURP, laser enucleation or vaporisation
    RADICAL_PROSTATECTOMY = "radical_prostatectomy"
    PROSTATE_BIOPSY = "prostate_biopsy"  # transrectal or transperineal
    URETHRAL_PROCEDURE = (
        "urethral_procedure"  # e.g. urethroplasty, urethrotomy, dilatation
    )
    INCONTINENCE_PROCEDURE = "incontinence_procedure"  # e.g. mid-urethral sling, colposuspension, artificial sphincter, bulking

    # Male genital organs
    ORCHIDECTOMY = "orchidectomy"
    ORCHIDOPEXY = "orchidopexy"
    SCROTAL_PROCEDURE = (
        "scrotal_procedure"  # e.g. exploration, hydrocele, epididymal cyst, varicocele
    )
    VASECTOMY = "vasectomy"
    VASECTOMY_REVERSAL = "vasectomy_reversal"
    CIRCUMCISION = "circumcision"
    PENILE_PROCEDURE = "penile_procedure"  # e.g. prosthesis, curvature correction

    # Female genital tract
    TOTAL_HYSTERECTOMY = "total_hysterectomy"  # abdominal, laparoscopic or vaginal
    SUBTOTAL_HYSTERECTOMY = "subtotal_hysterectomy"
    RADICAL_HYSTERECTOMY = "radical_hysterectomy"
    SALPINGO_OOPHORECTOMY = "salpingo_oophorectomy"  # unilateral or bilateral
    OVARIAN_CYSTECTOMY = "ovarian_cystectomy"
    SALPINGECTOMY = "salpingectomy"
    SALPINGOTOMY = "salpingotomy"
    MYOMECTOMY = "myomectomy"
    OMENTECTOMY = "omentectomy"
    CYTOREDUCTIVE_SURGERY = "cytoreductive_surgery"  # peritoneal debulking
    PELVIC_EXENTERATION = "pelvic_exenteration"
    HYSTEROSCOPY = "hysteroscopy"  # diagnostic or operative
    ENDOMETRIAL_ABLATION = "endometrial_ablation"
    UTERINE_CURETTAGE_OR_EVACUATION = (
        "uterine_curettage_or_evacuation"  # incl. D&C, evacuation of retained products
    )
    TUBAL_STERILISATION = "tubal_sterilisation"
    EXCISION_OF_CERVIX = "excision_of_cervix"  # cone biopsy or LLETZ
    VAGINAL_WALL_REPAIR = "vaginal_wall_repair"  # anterior or posterior
    SACROCOLPOPEXY = "sacrocolpopexy"
    VULVECTOMY = "vulvectomy"

    # Obstetric
    CAESAREAN_SECTION = "caesarean_section"
    INSTRUMENTAL_DELIVERY = "instrumental_delivery"  # forceps or ventouse
    REPAIR_OF_OBSTETRIC_PERINEAL_TEAR = "repair_of_obstetric_perineal_tear"
    MANUAL_REMOVAL_OF_PLACENTA = "manual_removal_of_placenta"
    CERVICAL_CERCLAGE = "cervical_cerclage"

    # Skin and soft tissue
    EXCISION_OF_SKIN_OR_SUBCUTANEOUS_LESION = "excision_of_skin_or_subcutaneous_lesion"  # incl. wide local excision, lipoma, cyst
    CURETTAGE_AND_CAUTERY_OF_SKIN_LESION = "curettage_and_cautery_of_skin_lesion"
    SKIN_GRAFT = "skin_graft"  # split or full thickness
    LOCAL_OR_REGIONAL_FLAP = "local_or_regional_flap"  # incl. pedicled flaps
    FREE_FLAP = "free_flap"  # microvascular free tissue transfer
    EXCISION_OF_SOFT_TISSUE_LESION = "excision_of_soft_tissue_lesion"  # deep to subcutaneous fat, e.g. ganglion, bursa, soft tissue tumour
    WOUND_WASHOUT_OR_DEBRIDEMENT = "wound_washout_or_debridement"
    NEGATIVE_PRESSURE_WOUND_THERAPY = "negative_pressure_wound_therapy"
    FASCIOTOMY = "fasciotomy"
    FASCIECTOMY = "fasciectomy"  # e.g. Dupuytren's
    SOFT_TISSUE_RELEASE = "soft_tissue_release"  # e.g. trigger finger, plantar fascia, tenolysis, joint contracture
    TENDON_REPAIR = "tendon_repair"  # any tendon, incl. Achilles and rotator cuff
    TENDON_TRANSFER = "tendon_transfer"
    TENDON_LENGTHENING_OR_TENOTOMY = "tendon_lengthening_or_tenotomy"
    MUSCLE_BIOPSY = "muscle_biopsy"
    MUSCLE_REPAIR = "muscle_repair"

    # Lymph nodes
    LYMPH_NODE_DISSECTION = "lymph_node_dissection"  # clearance of any nodal basin other than neck dissection
    SENTINEL_LYMPH_NODE_BIOPSY = "sentinel_lymph_node_biopsy"
    EXCISION_BIOPSY_OF_LYMPH_NODE = "excision_biopsy_of_lymph_node"

    # Interventional radiology (general)
    PERCUTANEOUS_BIOPSY = "percutaneous_biopsy"  # core or needle biopsy of any site without its own biopsy value
    IMAGE_GUIDED_DRAINAGE_OF_COLLECTION = "image_guided_drainage_of_collection"
    TUMOUR_ABLATION = (
        "tumour_ablation"  # any site or energy, percutaneous or intraoperative
    )
    THERAPEUTIC_EMBOLISATION = "therapeutic_embolisation"  # incl. chemoembolisation
    NERVE_BLOCK_OR_SPINAL_INJECTION = "nerve_block_or_spinal_injection"  # standalone pain procedure; not a block given as the anaesthetic

    # Skull, face and spine
    CRANIOPLASTY = "cranioplasty"
    ORTHOGNATHIC_SURGERY = "orthognathic_surgery"
    TEMPOROMANDIBULAR_JOINT_PROCEDURE = "temporomandibular_joint_procedure"
    SPINAL_DECOMPRESSION = (
        "spinal_decompression"  # laminectomy, laminotomy or foraminotomy
    )
    DISCECTOMY = "discectomy"  # any level or approach
    SPINAL_FUSION = "spinal_fusion"  # any level or approach, instrumented or not, incl. ACDF, interbody, deformity correction
    DISC_REPLACEMENT = "disc_replacement"
    VERTEBRAL_AUGMENTATION = "vertebral_augmentation"  # vertebroplasty or kyphoplasty
    EXCISION_OF_SPINAL_LESION = "excision_of_spinal_lesion"  # vertebral or intradural

    # Bones and joints - fracture
    OPEN_REDUCTION_INTERNAL_FIXATION = "open_reduction_internal_fixation"  # any bone, e.g. plate, screw, sliding hip screw
    CLOSED_REDUCTION_INTERNAL_FIXATION = "closed_reduction_internal_fixation"  # e.g. K-wires, percutaneous or cannulated screws
    INTRAMEDULLARY_NAILING = "intramedullary_nailing"
    EXTERNAL_FIXATION = "external_fixation"  # incl. circular frames
    CLOSED_REDUCTION_OF_FRACTURE = (
        "closed_reduction_of_fracture"  # manipulation without fixation
    )
    SKELETAL_TRACTION = "skeletal_traction"
    REMOVAL_OF_METALWORK = "removal_of_metalwork"

    # Bones and joints - arthroplasty
    TOTAL_HIP_REPLACEMENT = "total_hip_replacement"
    HIP_HEMIARTHROPLASTY = "hip_hemiarthroplasty"
    HIP_RESURFACING = "hip_resurfacing"
    TOTAL_KNEE_REPLACEMENT = "total_knee_replacement"
    PARTIAL_KNEE_REPLACEMENT = (
        "partial_knee_replacement"  # unicompartmental or patellofemoral
    )
    SHOULDER_REPLACEMENT = (
        "shoulder_replacement"  # anatomic, reverse or hemiarthroplasty
    )
    REVISION_ARTHROPLASTY = "revision_arthroplasty"  # any joint
    EXCISION_OR_INTERPOSITION_ARTHROPLASTY = "excision_or_interposition_arthroplasty"
    ENDOPROSTHETIC_REPLACEMENT = (
        "endoprosthetic_replacement"  # bone segment replacement
    )

    # Bones and joints - other
    EXCISION_OR_CURETTAGE_OF_BONE_LESION = "excision_or_curettage_of_bone_lesion"
    OSTEOTOMY = "osteotomy"  # any site
    BONE_GRAFTING = "bone_grafting"
    ARTHRODESIS = "arthrodesis"  # any joint
    REDUCTION_OF_JOINT_DISLOCATION = "reduction_of_joint_dislocation"  # open or closed
    SYNOVECTOMY = "synovectomy"
    MENISCAL_PROCEDURE = "meniscal_procedure"  # meniscectomy or repair
    CARTILAGE_PROCEDURE = "cartilage_procedure"
    LIGAMENT_RECONSTRUCTION_OR_REPAIR = (
        "ligament_reconstruction_or_repair"  # any ligament, incl. ACL
    )
    SHOULDER_STABILISATION = "shoulder_stabilisation"  # incl. labral repair, Latarjet
    SUBACROMIAL_DECOMPRESSION = "subacromial_decompression"
    JOINT_WASHOUT = "joint_washout"  # joint opened or scoped and irrigated
    JOINT_ASPIRATION_OR_INJECTION = "joint_aspiration_or_injection"
    MANIPULATION_UNDER_ANAESTHESIA_OF_JOINT = "manipulation_under_anaesthesia_of_joint"

    # Amputation
    MAJOR_LIMB_AMPUTATION = (
        "major_limb_amputation"  # at or proximal to the ankle or wrist
    )
    MINOR_AMPUTATION = "minor_amputation"  # digit, ray or partial foot or hand
    REVISION_OF_AMPUTATION_STUMP = "revision_of_amputation_stump"
    REPLANTATION_OF_LIMB_OR_DIGIT = "replantation_of_limb_or_digit"

    # Transplant organ retrieval
    ORGAN_RETRIEVAL = "organ_retrieval"  # live or deceased donor

    # General / cross-specialty
    DIAGNOSTIC_OR_EXPLORATORY_PROCEDURE = "diagnostic_or_exploratory_procedure"  # access and inspection only, with or without biopsy or washings, e.g. staging laparoscopy, EUA, diagnostic arthroscopy, open and close
    INCISION_AND_DRAINAGE_OF_ABSCESS = "incision_and_drainage_of_abscess"  # any site
    REMOVAL_OF_FOREIGN_BODY = "removal_of_foreign_body"


# ENUMS - COMPLICATIONS


class ComplicationType(str, Enum):
    """Type of complication of an operation.
    Use OTHER with complication_desc for a complication not listed here."""

    OTHER = "other"

    # Intraoperative - injury
    HAEMORRHAGE = "haemorrhage"  # major or unexpected bleeding, incl. requiring transfusion or post-operative bleeding
    VASCULAR_INJURY = "vascular_injury"
    NERVE_INJURY = "nerve_injury"
    BOWEL_INJURY_OR_ENTEROTOMY = "bowel_injury_or_enterotomy"
    BLADDER_OR_URETERIC_INJURY = "bladder_or_ureteric_injury"
    BILE_DUCT_INJURY = "bile_duct_injury"
    VISCERAL_ORGAN_INJURY = "visceral_organ_injury"
    DURAL_TEAR_OR_CSF_LEAK = "dural_tear_or_csf_leak"
    PNEUMOTHORAX = "pneumothorax"
    TENDON_OR_LIGAMENT_INJURY = "tendon_or_ligament_injury"
    IATROGENIC_FRACTURE = "iatrogenic_fracture"
    PERIPROSTHETIC_FRACTURE = "periprosthetic_fracture"
    SPILLAGE_OF_CONTENTS = (
        "spillage_of_contents"  # e.g. bile or stone spillage, tumour or cyst rupture
    )

    # Intraoperative - systemic / anaesthetic
    ANAESTHETIC_COMPLICATION = "anaesthetic_complication"
    DIFFICULT_OR_FAILED_AIRWAY = "difficult_or_failed_airway"
    ANAPHYLAXIS_ALLERGIC_REACTION = "anaphylaxis_allergic_reaction"
    CARDIOVASCULAR_COLLAPSE = (
        "cardiovascular_collapse"  # e.g. severe hypotension or shock
    )
    RESPIRATORY_FAILURE = (
        "respiratory_failure"  # e.g. severe hypoxia, unplanned ventilation
    )
    CARDIAC_ARREST = "cardiac_arrest"
    FAT_EMBOLISM_SYNDROME = "fat_embolism_syndrome"
    TOURNIQUET_RELATED_COMPLICATION = "tourniquet_related_complication"
    COMPARTMENT_SYNDROME = "compartment_syndrome"

    # Intraoperative - process / equipment
    EQUIPMENT_INSTRUMENT_FAILURE = "equipment_instrument_failure"
    WRONG_SITE_OR_PROCEDURE = "wrong_site_or_procedure"
    RETAINED_SURGICAL_ITEM = "retained_surgical_item"
    MEDICATION_ADMINISTRATION_ERROR = "medication_administration_error"
    BREACH_OF_STERILITY = "breach_of_sterility"

    # Implant-related
    IMPLANT_OR_GUIDEWIRE_MALPOSITION = "implant_or_guidewire_malposition"
    IMPLANT_FAILURE_LOOSENING_OR_BREAKAGE = "implant_failure_loosening_or_breakage"
    DISLOCATION_OR_INSTABILITY = "dislocation_or_instability"
    LEG_LENGTH_DISCREPANCY = "leg_length_discrepancy"

    # Postoperative (mainly for previous operations)
    ANASTOMOTIC_LEAK = "anastomotic_leak"
    SURGICAL_SITE_INFECTION = "surgical_site_infection"  # superficial, deep or organ space, incl. implant infection
    POSTOPERATIVE_COLLECTION = "postoperative_collection"  # abscess, haematoma, seroma
    WOUND_DEHISCENCE = "wound_dehiscence"


# ENUMS - SUPPORTING


class SurgicalApproach(str, Enum):
    """Access route through which the procedure was performed."""

    # Open
    LAPAROTOMY = "laparotomy"
    THORACOTOMY_OR_STERNOTOMY = "thoracotomy_or_sternotomy"
    CRANIOTOMY_OR_CRANIECTOMY = "craniotomy_or_craniectomy"
    BURR_HOLE = "burr_hole"
    OPEN_OTHER = "open_other"  # any other open incision, e.g. groin, neck, limb

    # Minimal access
    LAPAROSCOPIC = "laparoscopic"  # incl. hand-assisted
    THORACOSCOPIC = "thoracoscopic"  # incl. VATS
    ARTHROSCOPIC = "arthroscopic"
    ENDOSCOPIC = "endoscopic"  # via a scope through a natural orifice or small incision, e.g. GI, bronchoscopic, cystoscopic, hysteroscopic, transanal, endonasal
    ROBOTIC = "robotic"  # any robot-assisted approach

    # Needle, wire or catheter
    PERCUTANEOUS = "percutaneous"  # needle, wire or drain based, incl. image-guided
    ENDOVASCULAR = "endovascular"  # catheter based within vessels or heart

    CONVERTED_TO_OPEN = (
        "converted_to_open"  # minimal access or robotic approach converted to open
    )
    OTHER = "other"


class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    NOT_APPLICABLE = "not_applicable"  # multifocal or multisite


class OperationOutcome(str, Enum):
    """Outcome of the operation against what was planned. Single-valued;
    intraoperative_death takes precedence."""

    COMPLETED_AS_PLANNED = "completed_as_planned"
    MORE_THAN_PLANNED = "more_than_planned"  # additional or more extensive procedure, e.g. extended resection, unplanned stoma, procedure for an unexpected finding
    LESS_THAN_PLANNED = "less_than_planned"  # planned procedure found not necessary, and less extensive procedure done
    DIFFERENT_TO_PLANNED = "different_to_planned"  # alternative procedure of similar extent, e.g. Hartmann's instead of anterior resection
    ABANDONED = "abandoned"  # planned procedure not performed, e.g. open and close for unresectable disease, aborted for instability
    INTRAOPERATIVE_DEATH = "intraoperative_death"


class ProcedureUrgency(str, Enum):
    ELECTIVE = "elective"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class AnaestheticType(str, Enum):
    GENERAL = "general"
    SPINAL = "spinal"
    EPIDURAL = "epidural"  # incl. caudal
    PERIPHERAL_NERVE_BLOCK = (
        "peripheral_nerve_block"  # incl. plexus and fascial plane blocks
    )
    LOCAL = "local"  # local anaesthetic as the main anaesthetic
    SEDATION = "sedation"


class PreviousOperationRelation(str, Enum):
    """How the current operation relates to a previous operation."""

    RETURN_TO_THEATRE_FOR_COMPLICATION = (
        "return_to_theatre_for_complication"  # unplanned reoperation
    )
    PLANNED_STAGED_PROCEDURE = (
        "planned_staged_procedure"  # e.g. second look, staged reconstruction
    )
    REVERSAL = "reversal"  # e.g. stoma closure, Hartmann's reversal
    REVISION = "revision"  # of an implant or previous reconstruction
    COMPLETION = (
        "completion"  # e.g. completion thyroidectomy, completion lymph node dissection
    )
    RE_EXCISION = "re_excision"  # e.g. for involved or close margins
    REMOVAL_OF_IMPLANT = "removal_of_implant"
    TREATMENT_OF_RECURRENCE = "treatment_of_recurrence"  # same condition has recurred
    CONTRALATERAL_OR_OTHER_SITE = (
        "contralateral_or_other_site"  # same condition, other side or site
    )
    OTHER = "other"


# BLOCKS


class Implant(BaseModel):
    """A permanent device or material left implanted at the end of the procedure
    (e.g. prosthesis, cement, mesh, plate, screw, stent, pacemaker). Excludes sutures,
    clips, staples, dressings, drains, catheters and temporary devices."""

    implant_desc: str = Field(
        description="Direct extract naming the implant/device (e.g. 'Exeter V40 cemented stem', 'DePuy Pinnacle acetabular shell', 'size 5 mesh')"
    )
    device_type: Optional[str] = Field(
        None,
        description="General category of device (e.g. 'femoral stem', 'plate', 'mesh', 'screw')",
    )
    manufacturer: Optional[str] = Field(
        None, description="Manufacturer of the implant, if stated"
    )
    size_or_specification: Optional[str] = Field(
        None,
        description="Size, offset, or other specification of the implant, if stated",
    )
    serial_or_lot_number: Optional[str] = Field(
        None, description="Serial or batch/lot number of the implant, if stated"
    )


class Procedure(BaseModel):
    """A single procedure performed during the operation."""

    procedure: ProcedureType = Field(
        description="Specific procedure performed. Use OTHER with procedure_name_desc if not in enum."
    )
    procedure_name_desc: Optional[str] = Field(
        None,
        description="Direct extract naming the procedure as documented. Required when procedure is OTHER",
    )
    approach: Optional[SurgicalApproach] = Field(
        None,
        description="Access route through which this procedure was performed",
    )
    laterality: Optional[Laterality] = Field(
        None,
        description="Laterality of this procedure. not_applicable where it spans multiple foci or sites; None where not stated or the structure is unpaired",
    )
    site_detail: Optional[str] = Field(
        None,
        description="Location detail not captured by procedure (e.g. 'L4/5', 'right index finger', 'segment VII', 'axilla')",
    )
    incision_desc: Optional[str] = Field(
        None,
        description="Direct extract describing the incision or ports used",
    )
    implants: Optional[List[Implant]] = Field(
        None, description="Permanent implants left in place by this procedure"
    )


class ProcedureComplication(BaseModel):
    """A complication of an operation."""

    complication_type: ComplicationType = Field(
        description="Type of complication. Use OTHER if not in enum."
    )
    complication_desc: str = Field(
        description="Direct extract describing the complication as documented"
    )
    management_desc: Optional[str] = Field(
        None,
        description="Direct extract of how the complication was managed, if stated",
    )


class ProcedureMetadata(BaseModel):
    """Operative metadata reported for the case. None where not documented -
    do not infer or estimate."""

    urgency: Optional[ProcedureUrgency] = Field(
        None, description="Elective, urgent, or emergency, if stated"
    )
    anaesthetic_types: Optional[List[AnaestheticType]] = Field(
        None,
        description="Types of anaesthetic used, if stated. Multi-valued (e.g. general + regional)",
    )
    operative_time_minutes: Optional[int] = Field(
        None, ge=0, description="Total operative/procedure time in minutes, if stated"
    )
    estimated_blood_loss_ml: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated blood loss in millilitres, only if a single numeric value is stated",
    )
    estimated_blood_loss_desc: Optional[str] = Field(
        None,
        description="Direct extract of estimated blood loss as documented (e.g. 'minimal', '<50ml', '500-700ml')",
    )
    tourniquet_time_minutes: Optional[int] = Field(
        None, ge=0, description="Tourniquet time in minutes, if stated"
    )
    asa_grade: Optional[int] = Field(
        None, ge=1, le=6, description="ASA physical status grade (1-6), if stated"
    )
    asa_grade_desc: Optional[str] = Field(
        None, description="Direct extract of ASA grade as documented (e.g. 'ASA 3E')"
    )
    senior_surgeon_grade: Optional[str] = Field(
        None,
        description="Grade of the most senior surgeon scrubbed or supervising in theatre, as documented (e.g. 'consultant', 'registrar')",
    )


class PreviousOperation(BaseModel):
    """A previous operation that the note explicitly relates to the current operation."""

    relations_to_current: List[PreviousOperationRelation] = Field(
        min_length=1,
        description="How the current operation relates to this previous operation. Multi-valued (e.g. a second-look washout for infection is both planned_staged_procedure and return_to_theatre_for_complication). Use OTHER if not in enum.",
    )
    relation_desc: str = Field(
        description="Direct extract describing the relationship to the previous operation"
    )
    previous_operation_year: Optional[Year] = Field(
        None, description="Year of the previous operation"
    )
    previous_operation_month: Optional[Month] = Field(
        None, description="Month of the previous operation"
    )
    procedures: Optional[List[Procedure]] = Field(
        None, description="Procedures performed in the previous operation"
    )
    complications: Optional[List[ProcedureComplication]] = Field(
        None, description="Complications reported for the previous operation"
    )


# FINAL MODEL


class OperationNote(BaseModel):
    is_operation_note: bool = Field(
        description="True only if the document is an operation note for a procedure performed on a specific patient"
    )
    operation_year: Optional[Year] = Field(None, description="Year of the operation")
    operation_month: Optional[Month] = Field(None, description="Month of the operation")
    indication_summary: Optional[str] = Field(
        None, description="Short summary of the indication for the operation"
    )
    previous_related_operations: Optional[List[PreviousOperation]] = Field(
        None,
        description="Previous operations the note explicitly relates to this operation (e.g. return to theatre, reversal, revision)",
    )
    procedures: Optional[List[Procedure]] = Field(
        None,
        description="All procedures performed, in the order documented",
    )
    complications: Optional[List[ProcedureComplication]] = Field(
        None,
        description="Complications occurring during, or in immediate relation to, this operation. None if no complications are documented",
    )
    operation_outcome: Optional[OperationOutcome] = Field(
        None,
        description="Outcome of the operation against what was planned, only if the note makes this explicit",
    )
    operation_outcome_desc: Optional[str] = Field(
        None,
        description="Direct extract describing the outcome against what was planned",
    )
    metadata: Optional[ProcedureMetadata] = Field(
        None, description="Operative metadata reported for the case"
    )
    findings_summary: Optional[str] = Field(
        None,
        description="Short summary of any other findings encountered during the operation; None if not an operation note",
    )
    operation_summary: Optional[str] = Field(
        None,
        description="Short summary of the operation and its outcome; None if not an operation note",
    )
    post_op_plan_summary: Optional[str] = Field(
        None,
        description="Short summary of the post-operative plan with essential facts only",
    )
