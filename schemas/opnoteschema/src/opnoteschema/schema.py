from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

# TYPES

Year = Annotated[int, Field(ge=1900, le=2100)]
Month = Annotated[int, Field(ge=1, le=12)]


# ENUMS - PROCEDURE


class ProcedureType(str, Enum):
    """Specific procedure performed. Grouped by body system, loosely following OPCS-4
    chapter order for readability"""

    OTHER = "other"

    # Nervous system
    CRANIOTOMY = "craniotomy"  # only when no listed procedure was performed through it
    CRANIECTOMY = "craniectomy"  # incl. decompressive craniectomy
    EXCISION_OF_BRAIN_TUMOUR = "excision_of_brain_tumour"
    BRAIN_BIOPSY = "brain_biopsy"  # stereotactic or open
    EVACUATION_OF_EXTRADURAL_HAEMATOMA = "evacuation_of_extradural_haematoma"
    EVACUATION_OF_SUBDURAL_HAEMATOMA = (
        "evacuation_of_subdural_haematoma"  # burr hole or craniotomy
    )
    EVACUATION_OF_INTRACEREBRAL_HAEMATOMA = "evacuation_of_intracerebral_haematoma"
    CLIPPING_OF_CEREBRAL_ANEURYSM = "clipping_of_cerebral_aneurysm"
    ENDOVASCULAR_TREATMENT_OF_CEREBRAL_ANEURYSM = (
        "endovascular_treatment_of_cerebral_aneurysm"  # coiling, flow diversion
    )
    INSERTION_OF_VENTRICULOPERITONEAL_SHUNT = "insertion_of_ventriculoperitoneal_shunt"
    REVISION_OF_VENTRICULOPERITONEAL_SHUNT = "revision_of_ventriculoperitoneal_shunt"
    REMOVAL_OF_VENTRICULOPERITONEAL_SHUNT = "removal_of_ventriculoperitoneal_shunt"
    EXTERNAL_VENTRICULAR_DRAIN_INSERTION = "external_ventricular_drain_insertion"
    INSERTION_OF_INTRACRANIAL_PRESSURE_MONITOR = (
        "insertion_of_intracranial_pressure_monitor"
    )
    ENDOSCOPIC_THIRD_VENTRICULOSTOMY = "endoscopic_third_ventriculostomy"
    INSERTION_OF_DEEP_BRAIN_STIMULATOR = "insertion_of_deep_brain_stimulator"
    VAGAL_NERVE_STIMULATOR_INSERTION = "vagal_nerve_stimulator_insertion"
    MICROVASCULAR_DECOMPRESSION_OF_CRANIAL_NERVE = (
        "microvascular_decompression_of_cranial_nerve"
    )
    TRIGEMINAL_NERVE_RHIZOTOMY = "trigeminal_nerve_rhizotomy"
    REPAIR_OF_DURAL_TEAR = "repair_of_dural_tear"
    REPAIR_OF_SPINA_BIFIDA = "repair_of_spina_bifida"
    EXCISION_OF_INTRADURAL_SPINAL_TUMOUR = "excision_of_intradural_spinal_tumour"
    LUMBAR_PUNCTURE = "lumbar_puncture"  # incl. intrathecal drug administration
    INSERTION_OF_INTRATHECAL_PUMP = "insertion_of_intrathecal_pump"
    PERIPHERAL_NERVE_REPAIR_OR_GRAFT = "peripheral_nerve_repair_or_graft"
    PERIPHERAL_NERVE_DECOMPRESSION_OTHER = (
        "peripheral_nerve_decompression_other"  # not carpal/cubital tunnel
    )
    CARPAL_TUNNEL_DECOMPRESSION = "carpal_tunnel_decompression"
    CUBITAL_TUNNEL_DECOMPRESSION = "cubital_tunnel_decompression"
    EXCISION_OF_PERIPHERAL_NERVE_TUMOUR = "excision_of_peripheral_nerve_tumour"
    SYMPATHECTOMY = "sympathectomy"

    # Endocrine and breast
    TOTAL_THYROIDECTOMY = "total_thyroidectomy"
    SUBTOTAL_THYROIDECTOMY = "subtotal_thyroidectomy"
    THYROID_LOBECTOMY = "thyroid_lobectomy"  # hemithyroidectomy
    COMPLETION_THYROIDECTOMY = "completion_thyroidectomy"
    PARATHYROIDECTOMY = "parathyroidectomy"
    ADRENALECTOMY = "adrenalectomy"
    EXCISION_OF_THYROGLOSSAL_CYST = "excision_of_thyroglossal_cyst"
    SIMPLE_MASTECTOMY = "simple_mastectomy"
    SKIN_SPARING_MASTECTOMY = "skin_sparing_mastectomy"
    NIPPLE_SPARING_MASTECTOMY = "nipple_sparing_mastectomy"
    WIDE_LOCAL_EXCISION_OF_BREAST = "wide_local_excision_of_breast"
    EXCISION_OF_BREAST_LUMP = "excision_of_breast_lump"
    BREAST_RECONSTRUCTION = (
        "breast_reconstruction"  # implant or flap based, following mastectomy
    )
    BREAST_AUGMENTATION = "breast_augmentation"
    MICRODOCHECTOMY = "microdochectomy"

    # Eye
    CATARACT_EXTRACTION = (
        "cataract_extraction"  # incl. lens implant at the same sitting
    )
    SECONDARY_INTRAOCULAR_LENS_INSERTION = "secondary_intraocular_lens_insertion"
    VITRECTOMY = "vitrectomy"  # not for retinal detachment repair
    REPAIR_OF_RETINAL_DETACHMENT = "repair_of_retinal_detachment"  # any technique
    OPHTHALMIC_LASER_PROCEDURE = (
        "ophthalmic_laser_procedure"  # e.g. YAG capsulotomy, retinal photocoagulation
    )
    INTRAVITREAL_INJECTION = "intravitreal_injection"
    TRABECULECTOMY = "trabeculectomy"
    INSERTION_OF_GLAUCOMA_DRAINAGE_DEVICE = "insertion_of_glaucoma_drainage_device"
    CORNEAL_GRAFT = "corneal_graft"  # penetrating or lamellar
    EXCISION_OF_PTERYGIUM = "excision_of_pterygium"
    STRABISMUS_CORRECTION_SURGERY = "strabismus_correction_surgery"
    EVISCERATION_OF_EYE = "evisceration_of_eye"
    ENUCLEATION_OF_EYE = "enucleation_of_eye"
    REPAIR_OF_PTOSIS = "repair_of_ptosis"
    CORRECTION_OF_ENTROPION_OR_ECTROPION = "correction_of_entropion_or_ectropion"

    # Ear
    MYRINGOTOMY_WITH_GROMMET_INSERTION = "myringotomy_with_grommet_insertion"
    TYMPANOPLASTY = "tympanoplasty"  # incl. myringoplasty
    MASTOIDECTOMY = "mastoidectomy"
    COCHLEAR_IMPLANT_INSERTION = "cochlear_implant_insertion"
    BONE_ANCHORED_HEARING_AID_INSERTION = "bone_anchored_hearing_aid_insertion"
    STAPEDECTOMY = "stapedectomy"
    PINNAPLASTY = "pinnaplasty"

    # Respiratory tract
    LOBECTOMY_OF_LUNG = "lobectomy_of_lung"
    PNEUMONECTOMY = "pneumonectomy"
    SEGMENTECTOMY_OF_LUNG = "segmentectomy_of_lung"
    WEDGE_RESECTION_OF_LUNG = "wedge_resection_of_lung"
    THORACOTOMY = (
        "thoracotomy"  # only when no listed procedure was performed through it
    )
    PLEURODESIS = "pleurodesis"
    PLEURECTOMY = "pleurectomy"
    DECORTICATION_OF_LUNG = "decortication_of_lung"
    INSERTION_OF_CHEST_DRAIN = "insertion_of_chest_drain"
    TRACHEOSTOMY = "tracheostomy"  # surgical or percutaneous
    BRONCHOSCOPY = (
        "bronchoscopy"  # diagnostic or therapeutic, incl. EBUS, biopsy, stenting
    )
    MEDIASTINOSCOPY = "mediastinoscopy"
    LARYNGECTOMY = "laryngectomy"
    SEPTOPLASTY = "septoplasty"
    FUNCTIONAL_ENDOSCOPIC_SINUS_SURGERY = "functional_endoscopic_sinus_surgery"

    # Mouth
    DENTAL_EXTRACTION = "dental_extraction"
    TONSILLECTOMY = "tonsillectomy"
    ADENOIDECTOMY = "adenoidectomy"
    ADENOTONSILLECTOMY = "adenotonsillectomy"
    REPAIR_OF_CLEFT_LIP = "repair_of_cleft_lip"
    REPAIR_OF_CLEFT_PALATE = "repair_of_cleft_palate"
    EXCISION_OF_ORAL_LESION = "excision_of_oral_lesion"
    GLOSSECTOMY = "glossectomy"
    PAROTIDECTOMY = "parotidectomy"
    EXCISION_OF_SUBMANDIBULAR_GLAND = "excision_of_submandibular_gland"
    DRAINAGE_OF_PERITONSILLAR_ABSCESS = "drainage_of_peritonsillar_abscess"
    FRENULOPLASTY = "frenuloplasty"

    # Upper digestive system
    OESOPHAGECTOMY = "oesophagectomy"  # incl. oesophagogastrectomy; any technique (e.g. Ivor Lewis, transhiatal)
    HELLER_MYOTOMY = "heller_myotomy"
    FUNDOPLICATION = "fundoplication"
    HIATUS_HERNIA_REPAIR = "hiatus_hernia_repair"
    REPAIR_OF_PERFORATED_PEPTIC_ULCER = (
        "repair_of_perforated_peptic_ulcer"  # gastric or duodenal
    )
    OVERSEW_OF_BLEEDING_PEPTIC_ULCER = "oversew_of_bleeding_peptic_ulcer"
    TOTAL_GASTRECTOMY = "total_gastrectomy"
    PARTIAL_GASTRECTOMY = "partial_gastrectomy"  # distal, subtotal or wedge
    GASTROJEJUNOSTOMY = "gastrojejunostomy"
    PYLOROPLASTY = "pyloroplasty"
    PYLOROMYOTOMY = "pyloromyotomy"
    SLEEVE_GASTRECTOMY = "sleeve_gastrectomy"
    ROUX_EN_Y_GASTRIC_BYPASS = "roux_en_y_gastric_bypass"
    GASTRIC_BANDING = "gastric_banding"
    REMOVAL_OF_GASTRIC_BAND = "removal_of_gastric_band"
    GASTROSTOMY_INSERTION = "gastrostomy_insertion"  # PEG, RIG or surgical
    UPPER_GI_ENDOSCOPY = "upper_gi_endoscopy"  # diagnostic or therapeutic (e.g. biopsy, haemostasis, banding, dilatation, stenting, EMR/ESD)

    # Lower digestive system
    RIGHT_HEMICOLECTOMY = "right_hemicolectomy"
    EXTENDED_RIGHT_HEMICOLECTOMY = "extended_right_hemicolectomy"
    TRANSVERSE_COLECTOMY = "transverse_colectomy"
    LEFT_HEMICOLECTOMY = "left_hemicolectomy"
    SIGMOID_COLECTOMY = "sigmoid_colectomy"
    SUBTOTAL_COLECTOMY = "subtotal_colectomy"
    TOTAL_COLECTOMY = "total_colectomy"
    PANPROCTOCOLECTOMY = "panproctocolectomy"
    HARTMANNS_PROCEDURE = "hartmanns_procedure"
    REVERSAL_OF_HARTMANNS_PROCEDURE = "reversal_of_hartmanns_procedure"
    ANTERIOR_RESECTION_OF_RECTUM = "anterior_resection_of_rectum"  # incl. TME
    ABDOMINOPERINEAL_RESECTION_OF_RECTUM = "abdominoperineal_resection_of_rectum"
    ILEOCAECAL_RESECTION = "ileocaecal_resection"
    SMALL_BOWEL_RESECTION = "small_bowel_resection"
    SMALL_BOWEL_STRICTUROPLASTY = "small_bowel_stricturoplasty"
    FORMATION_OF_LOOP_ILEOSTOMY = "formation_of_loop_ileostomy"
    FORMATION_OF_END_ILEOSTOMY = "formation_of_end_ileostomy"
    FORMATION_OF_LOOP_COLOSTOMY = "formation_of_loop_colostomy"
    FORMATION_OF_END_COLOSTOMY = "formation_of_end_colostomy"
    CLOSURE_OF_ILEOSTOMY = "closure_of_ileostomy"
    CLOSURE_OF_COLOSTOMY = "closure_of_colostomy"
    ILEOANAL_POUCH_FORMATION = "ileoanal_pouch_formation"
    APPENDICECTOMY = "appendicectomy"
    ADHESIOLYSIS = "adhesiolysis"
    LOWER_GI_ENDOSCOPY = "lower_gi_endoscopy"  # colonoscopy or sigmoidoscopy, diagnostic or therapeutic (e.g. polypectomy, stenting)
    HAEMORRHOIDECTOMY = "haemorrhoidectomy"
    STAPLED_HAEMORRHOIDOPEXY = "stapled_haemorrhoidopexy"
    LATERAL_SPHINCTEROTOMY = "lateral_sphincterotomy"
    EXAMINATION_UNDER_ANAESTHESIA_OF_ANUS = "examination_under_anaesthesia_of_anus"
    DRAINAGE_OF_PERIANAL_ABSCESS = "drainage_of_perianal_abscess"
    FISTULOTOMY = "fistulotomy"
    INSERTION_OF_SETON = "insertion_of_seton"
    EXCISION_OF_PILONIDAL_SINUS = "excision_of_pilonidal_sinus"
    INGUINAL_HERNIA_REPAIR = "inguinal_hernia_repair"  # open, TEP or TAPP
    FEMORAL_HERNIA_REPAIR = "femoral_hernia_repair"
    UMBILICAL_HERNIA_REPAIR = "umbilical_hernia_repair"  # incl. paraumbilical
    EPIGASTRIC_HERNIA_REPAIR = "epigastric_hernia_repair"
    INCISIONAL_HERNIA_REPAIR = "incisional_hernia_repair"
    PARASTOMAL_HERNIA_REPAIR = "parastomal_hernia_repair"
    VENTRAL_HERNIA_REPAIR = (
        "ventral_hernia_repair"  # other ventral hernias not listed above
    )

    # Other abdominal organs
    CHOLECYSTECTOMY = "cholecystectomy"
    SUBTOTAL_CHOLECYSTECTOMY = "subtotal_cholecystectomy"
    CHOLECYSTOSTOMY = "cholecystostomy"
    BILE_DUCT_EXPLORATION = "bile_duct_exploration"
    HEPATICOJEJUNOSTOMY = "hepaticojejunostomy"
    ERCP = "ercp"  # diagnostic or therapeutic (e.g. sphincterotomy, stone extraction, stenting)
    PERCUTANEOUS_TRANSHEPATIC_BILIARY_PROCEDURE = (
        "percutaneous_transhepatic_biliary_procedure"  # PTC, biliary drain or stent
    )
    RIGHT_HEPATECTOMY = "right_hepatectomy"
    LEFT_HEPATECTOMY = "left_hepatectomy"
    LIVER_SEGMENTECTOMY = (
        "liver_segmentectomy"  # one or more segments, incl. left lateral sectionectomy
    )
    LIVER_WEDGE_RESECTION = "liver_wedge_resection"  # non-anatomical resection
    LIVER_TRANSPLANTATION = "liver_transplantation"
    PANCREATICODUODENECTOMY = (
        "pancreaticoduodenectomy"  # Whipple's or pylorus-preserving
    )
    DISTAL_PANCREATECTOMY = "distal_pancreatectomy"
    TOTAL_PANCREATECTOMY = "total_pancreatectomy"
    PANCREATIC_NECROSECTOMY = "pancreatic_necrosectomy"
    SPLENECTOMY = "splenectomy"
    PARTIAL_SPLENECTOMY = "partial_splenectomy"

    # Heart
    CORONARY_ARTERY_BYPASS_GRAFT = "coronary_artery_bypass_graft"  # on or off pump
    CORONARY_ANGIOGRAPHY = "coronary_angiography"  # diagnostic cardiac catheterisation
    PERCUTANEOUS_CORONARY_INTERVENTION = (
        "percutaneous_coronary_intervention"  # angioplasty with or without stent
    )
    AORTIC_VALVE_REPLACEMENT = "aortic_valve_replacement"
    TRANSCATHETER_AORTIC_VALVE_IMPLANTATION = "transcatheter_aortic_valve_implantation"
    MITRAL_VALVE_REPLACEMENT = "mitral_valve_replacement"
    MITRAL_VALVE_REPAIR = "mitral_valve_repair"
    TRICUSPID_VALVE_REPAIR = "tricuspid_valve_repair"
    TRICUSPID_VALVE_REPLACEMENT = "tricuspid_valve_replacement"
    PULMONARY_VALVE_REPLACEMENT = "pulmonary_valve_replacement"
    CARDIAC_DEVICE_INSERTION_OR_CHANGE = (
        "cardiac_device_insertion_or_change"  # pacemaker, ICD, CRT, generator change
    )
    CARDIAC_DEVICE_LEAD_EXTRACTION = "cardiac_device_lead_extraction"
    CATHETER_ABLATION_OR_ELECTROPHYSIOLOGY_STUDY = (
        "catheter_ablation_or_electrophysiology_study"  # incl. pulmonary vein isolation
    )
    MAZE_PROCEDURE = "maze_procedure"  # surgical ablation
    PERICARDIOCENTESIS = "pericardiocentesis"
    PERICARDIAL_WINDOW = "pericardial_window"
    CARDIAC_TRANSPLANTATION = "cardiac_transplantation"
    LEFT_VENTRICULAR_ASSIST_DEVICE_INSERTION = (
        "left_ventricular_assist_device_insertion"
    )
    INTRA_AORTIC_BALLOON_PUMP_INSERTION = "intra_aortic_balloon_pump_insertion"
    EXTRACORPOREAL_MEMBRANE_OXYGENATION_CANNULATION = (
        "extracorporeal_membrane_oxygenation_cannulation"
    )
    ATRIAL_SEPTAL_DEFECT_REPAIR = (
        "atrial_septal_defect_repair"  # surgical or device closure
    )
    VENTRICULAR_SEPTAL_DEFECT_REPAIR = "ventricular_septal_defect_repair"
    CLOSURE_OF_PATENT_DUCTUS_ARTERIOSUS = "closure_of_patent_ductus_arteriosus"

    # Arteries and veins
    ABDOMINAL_AORTIC_ANEURYSM_REPAIR = (
        "abdominal_aortic_aneurysm_repair"  # open or endovascular (EVAR)
    )
    THORACIC_AORTIC_ANEURYSM_REPAIR = (
        "thoracic_aortic_aneurysm_repair"  # open or endovascular (TEVAR)
    )
    AORTIC_DISSECTION_REPAIR = "aortic_dissection_repair"
    CAROTID_ENDARTERECTOMY = "carotid_endarterectomy"
    CAROTID_ARTERY_STENTING = "carotid_artery_stenting"
    FEMOROPOPLITEAL_BYPASS = "femoropopliteal_bypass"
    FEMORODISTAL_BYPASS = "femorodistal_bypass"
    FEMOROFEMORAL_CROSSOVER_BYPASS = "femorofemoral_crossover_bypass"
    AXILLOFEMORAL_BYPASS = "axillofemoral_bypass"
    AORTOFEMORAL_BYPASS = "aortofemoral_bypass"  # uni- or bifemoral
    PERIPHERAL_ANGIOPLASTY_OR_STENTING = (
        "peripheral_angioplasty_or_stenting"  # non-coronary, non-carotid
    )
    ARTERIAL_EMBOLECTOMY_OR_THROMBECTOMY = "arterial_embolectomy_or_thrombectomy"
    VENOUS_THROMBECTOMY = "venous_thrombectomy"
    CREATION_OF_ARTERIOVENOUS_FISTULA = "creation_of_arteriovenous_fistula"
    INSERTION_OF_ARTERIOVENOUS_GRAFT = "insertion_of_arteriovenous_graft"
    VARICOSE_VEIN_PROCEDURE = "varicose_vein_procedure"  # any technique (ligation, stripping, phlebectomy, endovenous ablation, foam)
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
    LIVE_DONOR_NEPHRECTOMY = "live_donor_nephrectomy"
    RENAL_TRANSPLANTATION = "renal_transplantation"
    PYELOPLASTY = "pyeloplasty"
    NEPHROSTOMY_INSERTION = "nephrostomy_insertion"
    PERCUTANEOUS_NEPHROLITHOTOMY = "percutaneous_nephrolithotomy"
    URETEROSCOPY = "ureteroscopy"  # diagnostic or therapeutic, incl. laser lithotripsy
    INSERTION_OF_URETERIC_STENT = "insertion_of_ureteric_stent"
    URETERIC_REIMPLANTATION = "ureteric_reimplantation"
    RADICAL_CYSTECTOMY = "radical_cystectomy"
    PARTIAL_CYSTECTOMY = "partial_cystectomy"
    ILEAL_CONDUIT_FORMATION = "ileal_conduit_formation"
    TRANSURETHRAL_RESECTION_OF_BLADDER_TUMOUR = (
        "transurethral_resection_of_bladder_tumour"
    )
    CYSTOSCOPY = "cystoscopy"  # diagnostic or therapeutic (e.g. diathermy, biopsy, stent removal)
    INSERTION_OF_SUPRAPUBIC_CATHETER = "insertion_of_suprapubic_catheter"
    TRANSURETHRAL_RESECTION_OF_PROSTATE = "transurethral_resection_of_prostate"
    LASER_ENUCLEATION_OF_PROSTATE = "laser_enucleation_of_prostate"
    RADICAL_PROSTATECTOMY = "radical_prostatectomy"
    PROSTATE_BIOPSY = "prostate_biopsy"  # transrectal or transperineal
    URETHROPLASTY = "urethroplasty"
    OPTICAL_URETHROTOMY = "optical_urethrotomy"
    URETHRAL_DILATATION = "urethral_dilatation"
    INSERTION_OF_ARTIFICIAL_URINARY_SPHINCTER = (
        "insertion_of_artificial_urinary_sphincter"
    )
    MID_URETHRAL_SLING_PROCEDURE = "mid_urethral_sling_procedure"
    COLPOSUSPENSION = "colposuspension"

    # Male genital organs
    ORCHIDECTOMY = "orchidectomy"
    ORCHIDOPEXY = "orchidopexy"
    SCROTAL_EXPLORATION = "scrotal_exploration"  # e.g. for suspected torsion, with or without detorsion and fixation
    VASECTOMY = "vasectomy"
    VASECTOMY_REVERSAL = "vasectomy_reversal"
    HYDROCELE_REPAIR = "hydrocele_repair"
    EPIDIDYMAL_CYST_EXCISION = "epididymal_cyst_excision"
    VARICOCELECTOMY = "varicocelectomy"
    CIRCUMCISION = "circumcision"
    PENILE_PROSTHESIS_INSERTION = "penile_prosthesis_insertion"
    CORRECTION_OF_PENILE_CURVATURE = (
        "correction_of_penile_curvature"  # e.g. Nesbit procedure
    )

    # Female genital tract
    TOTAL_HYSTERECTOMY = "total_hysterectomy"  # abdominal, laparoscopic or vaginal
    SUBTOTAL_HYSTERECTOMY = "subtotal_hysterectomy"
    RADICAL_HYSTERECTOMY = "radical_hysterectomy"
    SALPINGO_OOPHORECTOMY = (
        "salpingo_oophorectomy"  # unilateral or bilateral via laterality
    )
    OVARIAN_CYSTECTOMY = "ovarian_cystectomy"
    SALPINGECTOMY = "salpingectomy"
    SALPINGOTOMY = "salpingotomy"
    MYOMECTOMY = "myomectomy"
    OMENTECTOMY = "omentectomy"
    CYTOREDUCTIVE_SURGERY = (
        "cytoreductive_surgery"  # debulking of peritoneal / ovarian malignancy
    )
    PELVIC_EXENTERATION = "pelvic_exenteration"
    ENDOMETRIAL_ABLATION = "endometrial_ablation"
    DILATATION_AND_CURETTAGE = "dilatation_and_curettage"
    HYSTEROSCOPY = "hysteroscopy"  # diagnostic or operative (e.g. polypectomy)
    TUBAL_STERILISATION = "tubal_sterilisation"
    CONE_BIOPSY_OF_CERVIX = "cone_biopsy_of_cervix"
    LOOP_EXCISION_OF_TRANSFORMATION_ZONE = "loop_excision_of_transformation_zone"
    VAGINAL_REPAIR_ANTERIOR = "vaginal_repair_anterior"
    VAGINAL_REPAIR_POSTERIOR = "vaginal_repair_posterior"
    SACROCOLPOPEXY = "sacrocolpopexy"
    VULVECTOMY = "vulvectomy"

    # Obstetric
    CAESAREAN_SECTION = "caesarean_section"  # lower segment or classical
    CAESAREAN_HYSTERECTOMY = "caesarean_hysterectomy"
    INSTRUMENTAL_DELIVERY = "instrumental_delivery"  # forceps or ventouse
    REPAIR_OF_OBSTETRIC_PERINEAL_TEAR = "repair_of_obstetric_perineal_tear"
    MANUAL_REMOVAL_OF_PLACENTA = "manual_removal_of_placenta"
    EVACUATION_OF_RETAINED_PRODUCTS_OF_CONCEPTION = (
        "evacuation_of_retained_products_of_conception"
    )
    CERVICAL_CERCLAGE = "cervical_cerclage"

    # Skin
    EXCISION_OF_SKIN_LESION = "excision_of_skin_lesion"
    WIDE_LOCAL_EXCISION_OF_SKIN_LESION = "wide_local_excision_of_skin_lesion"
    CURETTAGE_AND_CAUTERY_OF_SKIN_LESION = "curettage_and_cautery_of_skin_lesion"
    SPLIT_SKIN_GRAFT = "split_skin_graft"
    FULL_THICKNESS_SKIN_GRAFT = "full_thickness_skin_graft"
    LOCAL_OR_REGIONAL_FLAP = "local_or_regional_flap"  # incl. pedicled flaps
    FREE_FLAP = "free_flap"  # microvascular free tissue transfer
    EXCISION_OF_LIPOMA = "excision_of_lipoma"
    EXCISION_OF_SEBACEOUS_CYST = "excision_of_sebaceous_cyst"
    NEGATIVE_PRESSURE_WOUND_THERAPY = "negative_pressure_wound_therapy"

    # Soft tissue and lymph nodes
    FASCIOTOMY = "fasciotomy"
    PLANTAR_FASCIA_RELEASE = "plantar_fascia_release"
    DUPUYTRENS_FASCIECTOMY = "dupuytrens_fasciectomy"
    EXCISION_OF_GANGLION = "excision_of_ganglion"
    BURSECTOMY = "bursectomy"
    EXCISION_OF_SOFT_TISSUE_TUMOUR = (
        "excision_of_soft_tissue_tumour"  # e.g. sarcoma; not lipoma
    )
    TENDON_REPAIR = (
        "tendon_repair"  # primary or secondary; not Achilles or rotator cuff
    )
    ACHILLES_TENDON_REPAIR = "achilles_tendon_repair"
    ROTATOR_CUFF_REPAIR = "rotator_cuff_repair"
    TENDON_TRANSFER = "tendon_transfer"
    TENOLYSIS = "tenolysis"
    TENDON_LENGTHENING_OR_TENOTOMY = "tendon_lengthening_or_tenotomy"
    EXCISION_OF_TENDON_OR_TENDON_SHEATH_LESION = (
        "excision_of_tendon_or_tendon_sheath_lesion"
    )
    TRIGGER_FINGER_RELEASE = "trigger_finger_release"
    MUSCLE_BIOPSY = "muscle_biopsy"
    MUSCLE_REPAIR = "muscle_repair"
    LYMPH_NODE_DISSECTION = "lymph_node_dissection"  # block dissection or clearance of any basin (e.g. axillary, neck, inguinal, pelvic)
    SENTINEL_LYMPH_NODE_BIOPSY = "sentinel_lymph_node_biopsy"
    EXCISION_BIOPSY_OF_LYMPH_NODE = "excision_biopsy_of_lymph_node"

    # Interventional radiology (general)
    PERCUTANEOUS_BIOPSY = "percutaneous_biopsy"  # core or needle biopsy of any site, image-guided or freehand
    IMAGE_GUIDED_DRAINAGE_OF_COLLECTION = "image_guided_drainage_of_collection"
    TUMOUR_ABLATION = "tumour_ablation"  # any site or energy (e.g. RFA, microwave, cryo), percutaneous or intraoperative
    THERAPEUTIC_EMBOLISATION = "therapeutic_embolisation"  # incl. chemoembolisation
    NERVE_BLOCK_OR_SPINAL_INJECTION = "nerve_block_or_spinal_injection"  # standalone pain procedure (incl. facet injection/denervation); not a block given as part of anaesthesia

    # Skull, face and spine
    CRANIOPLASTY = "cranioplasty"
    ORTHOGNATHIC_SURGERY = "orthognathic_surgery"  # e.g. Le Fort, mandibular osteotomy
    FIXATION_OF_FACIAL_FRACTURE = (
        "fixation_of_facial_fracture"  # mandible, zygoma, maxilla, orbit
    )
    TEMPOROMANDIBULAR_JOINT_REPLACEMENT = "temporomandibular_joint_replacement"
    SPINAL_DECOMPRESSION = (
        "spinal_decompression"  # laminectomy, laminotomy or foraminotomy, any level
    )
    DISCECTOMY = "discectomy"  # any level; not ACDF
    ANTERIOR_CERVICAL_DISCECTOMY_AND_FUSION = "anterior_cervical_discectomy_and_fusion"
    DISC_REPLACEMENT = "disc_replacement"
    POSTERIOR_SPINAL_FUSION = (
        "posterior_spinal_fusion"  # instrumented or non-instrumented, any level
    )
    INTERBODY_FUSION = "interbody_fusion"  # e.g. TLIF, PLIF, ALIF, lateral
    SPINAL_DEFORMITY_CORRECTION = "spinal_deformity_correction"  # e.g. scoliosis
    INTERSPINOUS_SPACER_INSERTION = "interspinous_spacer_insertion"
    EXCISION_OF_VERTEBRAL_LESION = "excision_of_vertebral_lesion"
    VERTEBRAL_AUGMENTATION = "vertebral_augmentation"  # vertebroplasty or kyphoplasty
    FIXATION_OF_SPINAL_FRACTURE = "fixation_of_spinal_fracture"
    SPINAL_CORD_STIMULATOR_INSERTION = "spinal_cord_stimulator_insertion"

    # Other bones and joints - fracture
    OPEN_REDUCTION_INTERNAL_FIXATION = "open_reduction_internal_fixation"  # plate/screw fixation of any fracture not listed below
    CLOSED_REDUCTION_INTERNAL_FIXATION = (
        "closed_reduction_internal_fixation"  # e.g. K-wires, percutaneous screws
    )
    INTRAMEDULLARY_NAILING = "intramedullary_nailing"  # any long bone
    DYNAMIC_HIP_SCREW_FIXATION = "dynamic_hip_screw_fixation"
    CANNULATED_SCREW_FIXATION_NECK_OF_FEMUR = "cannulated_screw_fixation_neck_of_femur"
    EXTERNAL_FIXATION = "external_fixation"  # incl. circular frames
    CLOSED_REDUCTION_OF_FRACTURE = (
        "closed_reduction_of_fracture"  # manipulation without fixation
    )
    SKELETAL_TRACTION = "skeletal_traction"
    REMOVAL_OF_METALWORK = "removal_of_metalwork"

    # Other bones and joints - arthroplasty
    TOTAL_HIP_REPLACEMENT = "total_hip_replacement"
    REVISION_TOTAL_HIP_REPLACEMENT = "revision_total_hip_replacement"
    HIP_HEMIARTHROPLASTY = "hip_hemiarthroplasty"
    HIP_RESURFACING = "hip_resurfacing"
    TOTAL_KNEE_REPLACEMENT = "total_knee_replacement"
    REVISION_TOTAL_KNEE_REPLACEMENT = "revision_total_knee_replacement"
    UNICOMPARTMENTAL_KNEE_REPLACEMENT = "unicompartmental_knee_replacement"
    PATELLOFEMORAL_JOINT_REPLACEMENT = "patellofemoral_joint_replacement"
    TOTAL_SHOULDER_REPLACEMENT = "total_shoulder_replacement"  # anatomic
    REVERSE_TOTAL_SHOULDER_REPLACEMENT = "reverse_total_shoulder_replacement"
    SHOULDER_HEMIARTHROPLASTY = "shoulder_hemiarthroplasty"
    TOTAL_ELBOW_REPLACEMENT = "total_elbow_replacement"
    TOTAL_ANKLE_REPLACEMENT = "total_ankle_replacement"
    RADIAL_HEAD_REPLACEMENT = "radial_head_replacement"
    EXCISION_ARTHROPLASTY = "excision_arthroplasty"
    INTERPOSITION_ARTHROPLASTY = "interposition_arthroplasty"
    ENDOPROSTHETIC_REPLACEMENT = "endoprosthetic_replacement"  # replacement of a bone segment, e.g. after tumour resection

    # Other bones and joints - other
    EXCISION_OF_BONE_TUMOUR = "excision_of_bone_tumour"
    CURETTAGE_OF_BONE_LESION = "curettage_of_bone_lesion"
    HIGH_TIBIAL_OSTEOTOMY = "high_tibial_osteotomy"
    PELVIC_OSTEOTOMY = "pelvic_osteotomy"
    HALLUX_VALGUS_CORRECTION = "hallux_valgus_correction"  # first metatarsal osteotomy
    OSTEOTOMY = "osteotomy"  # other sites
    BONE_GRAFTING = "bone_grafting"
    BONE_MARROW_ASPIRATION_OR_BIOPSY = "bone_marrow_aspiration_or_biopsy"
    ANKLE_ARTHRODESIS = "ankle_arthrodesis"
    FIRST_MTP_JOINT_FUSION = "first_mtp_joint_fusion"
    ARTHRODESIS = "arthrodesis"  # other joints
    OPEN_REDUCTION_OF_JOINT_DISLOCATION = "open_reduction_of_joint_dislocation"
    CLOSED_REDUCTION_OF_JOINT_DISLOCATION = "closed_reduction_of_joint_dislocation"
    SYNOVECTOMY = "synovectomy"
    MENISCECTOMY = "meniscectomy"
    MENISCAL_REPAIR = "meniscal_repair"
    KNEE_CARTILAGE_PROCEDURE = "knee_cartilage_procedure"
    ACL_RECONSTRUCTION = "acl_reconstruction"
    LIGAMENT_RECONSTRUCTION = "ligament_reconstruction"  # other ligaments
    LIGAMENT_REPAIR = "ligament_repair"
    SHOULDER_STABILISATION = "shoulder_stabilisation"  # incl. labral repair, Latarjet
    SUBACROMIAL_DECOMPRESSION = "subacromial_decompression"
    RELEASE_OF_JOINT_CONTRACTURE = "release_of_joint_contracture"
    JOINT_WASHOUT = "joint_washout"
    DIAGNOSTIC_ARTHROSCOPY = (
        "diagnostic_arthroscopy"  # any joint, no other procedure performed
    )
    JOINT_ASPIRATION_OR_INJECTION = "joint_aspiration_or_injection"
    MANIPULATION_UNDER_ANAESTHESIA_OF_JOINT = "manipulation_under_anaesthesia_of_joint"

    # Amputation
    AMPUTATION_ABOVE_KNEE = "amputation_above_knee"
    AMPUTATION_BELOW_KNEE = "amputation_below_knee"
    AMPUTATION_OF_TOE = "amputation_of_toe"
    AMPUTATION_OF_FINGER = "amputation_of_finger"
    AMPUTATION = "amputation"  # other levels
    REVISION_OF_AMPUTATION_STUMP = "revision_of_amputation_stump"
    REPLANTATION_OF_LIMB_OR_DIGIT = "replantation_of_limb_or_digit"

    # General / cross-specialty
    DIAGNOSTIC_LAPAROSCOPY = "diagnostic_laparoscopy"  # no other procedure performed
    EXPLORATORY_LAPAROTOMY = "exploratory_laparotomy"  # only when no listed procedure was performed through it
    WOUND_DEBRIDEMENT = "wound_debridement"
    INCISION_AND_DRAINAGE_OF_ABSCESS = (
        "incision_and_drainage_of_abscess"  # sites not listed elsewhere
    )
    REMOVAL_OF_FOREIGN_BODY = "removal_of_foreign_body"
    EXAMINATION_UNDER_ANAESTHESIA = (
        "examination_under_anaesthesia"  # sites not listed elsewhere
    )


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
    VISCERAL_ORGAN_INJURY = "visceral_organ_injury"  # organs not listed above
    DURAL_TEAR_OR_CSF_LEAK = "dural_tear_or_csf_leak"
    PNEUMOTHORAX = "pneumothorax"
    POSTERIOR_CAPSULE_RUPTURE = "posterior_capsule_rupture"
    TENDON_OR_LIGAMENT_INJURY = "tendon_or_ligament_injury"
    IATROGENIC_FRACTURE = "iatrogenic_fracture"  # not periprosthetic
    PERIPROSTHETIC_FRACTURE = "periprosthetic_fracture"
    SPILLAGE_OF_CONTENTS = (
        "spillage_of_contents"  # e.g. bile or stone spillage, tumour or cyst rupture
    )

    # Intraoperative - systemic / anaesthetic
    ANAESTHETIC_COMPLICATION = "anaesthetic_complication"  # other than airway
    DIFFICULT_OR_FAILED_AIRWAY = "difficult_or_failed_airway"
    ANAPHYLAXIS_ALLERGIC_REACTION = "anaphylaxis_allergic_reaction"
    CARDIAC_ARREST = "cardiac_arrest"
    INTRAOPERATIVE_DEATH = "intraoperative_death"
    BONE_CEMENT_IMPLANTATION_SYNDROME = "bone_cement_implantation_syndrome"
    FAT_EMBOLISM_SYNDROME = "fat_embolism_syndrome"
    TOURNIQUET_RELATED_COMPLICATION = "tourniquet_related_complication"
    COMPARTMENT_SYNDROME = "compartment_syndrome"

    # Intraoperative - process / equipment
    EQUIPMENT_INSTRUMENT_FAILURE = "equipment_instrument_failure"
    WRONG_SITE_OR_PROCEDURE = "wrong_site_or_procedure"
    RETAINED_SURGICAL_ITEM = "retained_surgical_item"
    MEDICATION_ADMINISTRATION_ERROR = "medication_administration_error"
    BREACH_OF_STERILITY = "breach_of_sterility"
    PROCEDURE_ABANDONED_OR_INCOMPLETE = "procedure_abandoned_or_incomplete"

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
    OPEN = "open"
    MINIMAL_ACCESS = "minimal_access"  # laparoscopic, thoracoscopic, arthroscopic, endoscopic; incl. hand-assisted/hybrid; not robotic
    ROBOTIC = "robotic"
    PERCUTANEOUS_OR_ENDOVASCULAR = (
        "percutaneous_or_endovascular"  # needle, wire or catheter based, incl. IR
    )
    CONVERTED_TO_OPEN = (
        "converted_to_open"  # minimal access or robotic approach converted to open
    )
    OTHER = "other"


class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    NOT_APPLICABLE = "not_applicable"


class ProcedureUrgency(str, Enum):
    ELECTIVE = "elective"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class AnaestheticType(str, Enum):
    GENERAL = "general"
    REGIONAL = "regional"  # spinal, epidural or nerve block
    LOCAL = "local"
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
    """A device or implant used or inserted during the procedure."""

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

    procedure_type: ProcedureType = Field(
        description="Specific procedure performed. Use OTHER with procedure_desc if not in enum."
    )
    procedure_desc: Optional[str] = Field(
        None,
        description="Direct extract naming the procedure as documented. Required when procedure_type is OTHER",
    )
    approach: Optional[SurgicalApproach] = Field(
        None,
        description="Surgical approach, including where implied by the procedure (e.g. ERCP is minimal_access)",
    )
    laterality: Optional[Laterality] = Field(
        None, description="Laterality of this procedure"
    )
    site_detail: Optional[str] = Field(
        None,
        description="Location detail not captured by procedure_type (e.g. 'L4/5', 'right index finger', 'segment VII', 'axilla')",
    )
    incision_desc: Optional[str] = Field(
        None,
        description="Direct extract describing the incision or ports used",
    )
    implants: Optional[List[Implant]] = Field(
        None, description="Implants or devices used in this procedure"
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


class Drain(BaseModel):
    """A drain left in place at the end of the operation."""

    drain_type: Optional[str] = Field(
        None, description="Type of drain (e.g. 'Robinson', 'Redivac', 'chest drain')"
    )
    drain_site: Optional[str] = Field(
        None, description="Where the drain is placed (e.g. 'pelvis', 'subhepatic')"
    )
    drain_desc: str = Field(description="Direct extract describing the drain")


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
    surgeon_grade: Optional[str] = Field(
        None,
        description="Grade or seniority of the operating surgeon as documented (e.g. 'consultant', 'registrar')",
    )


class PreviousOperation(BaseModel):
    """A previous operation that the note explicitly relates to the current operation."""

    relation_to_current: PreviousOperationRelation = Field(
        description="How the current operation relates to this previous operation. Use OTHER if not in enum."
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
        description="True only if the document records a surgical, endoscopic or interventional procedure performed on a specific patient"
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
        None, description="All procedures performed, in the order documented"
    )
    complications: Optional[List[ProcedureComplication]] = Field(
        None,
        description="Complications occurring during, or in immediate relation to, this operation; None if none reported",
    )
    drains: Optional[List[Drain]] = Field(
        None, description="Drains left in place at the end of the operation"
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
