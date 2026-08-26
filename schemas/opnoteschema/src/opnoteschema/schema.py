from enum import Enum

from pydantic import BaseModel, Field


# ENUMS - PROCEDURE


class ProcedureType(str, Enum):
    """Specific procedure performed. Grouped by OPCS-4 chapter for readability.
    Use OTHER with procedure_desc for a procedure not listed here."""

    OTHER = "other"

    # A Nervous System
    CRANIOTOMY = "craniotomy"
    CRANIECTOMY = "craniectomy"
    BURR_HOLE_DRAINAGE = "burr_hole_drainage"
    EXCISION_OF_BRAIN_TUMOUR = "excision_of_brain_tumour"
    STEREOTACTIC_BRAIN_BIOPSY = "stereotactic_brain_biopsy"
    STEREOTACTIC_RADIOSURGERY = "stereotactic_radiosurgery"
    EVACUATION_OF_EXTRADURAL_HAEMATOMA = "evacuation_of_extradural_haematoma"
    EVACUATION_OF_SUBDURAL_HAEMATOMA = "evacuation_of_subdural_haematoma"
    EVACUATION_OF_INTRACEREBRAL_HAEMATOMA = "evacuation_of_intracerebral_haematoma"
    DECOMPRESSIVE_CRANIECTOMY = "decompressive_craniectomy"
    CLIPPING_OF_CEREBRAL_ANEURYSM = "clipping_of_cerebral_aneurysm"
    ENDOVASCULAR_COILING_OF_CEREBRAL_ANEURYSM = (
        "endovascular_coiling_of_cerebral_aneurysm"
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
    SPINAL_CORD_TUMOUR_EXCISION = "spinal_cord_tumour_excision"
    LUMBAR_PUNCTURE = "lumbar_puncture"
    INSERTION_OF_INTRATHECAL_PUMP = "insertion_of_intrathecal_pump"
    PERIPHERAL_NERVE_GRAFT = "peripheral_nerve_graft"
    EXCISION_OF_PERIPHERAL_NERVE_TUMOUR = "excision_of_peripheral_nerve_tumour"
    SYMPATHECTOMY = "sympathectomy"

    # B Endocrine System and Breast
    TOTAL_THYROIDECTOMY = "total_thyroidectomy"
    SUBTOTAL_THYROIDECTOMY = "subtotal_thyroidectomy"
    THYROID_LOBECTOMY = "thyroid_lobectomy"
    COMPLETION_THYROIDECTOMY = "completion_thyroidectomy"
    PARATHYROIDECTOMY = "parathyroidectomy"
    ADRENALECTOMY = "adrenalectomy"
    LAPAROSCOPIC_ADRENALECTOMY = "laparoscopic_adrenalectomy"
    EXCISION_OF_THYROGLOSSAL_CYST = "excision_of_thyroglossal_cyst"
    SIMPLE_MASTECTOMY = "simple_mastectomy"
    SKIN_SPARING_MASTECTOMY = "skin_sparing_mastectomy"
    NIPPLE_SPARING_MASTECTOMY = "nipple_sparing_mastectomy"
    BILATERAL_MASTECTOMY = "bilateral_mastectomy"
    WIDE_LOCAL_EXCISION_OF_BREAST_LESION = "wide_local_excision_of_breast_lesion"
    BREAST_CONSERVING_SURGERY = "breast_conserving_surgery"
    AXILLARY_LYMPH_NODE_CLEARANCE = "axillary_lymph_node_clearance"
    SENTINEL_LYMPH_NODE_BIOPSY_BREAST = "sentinel_lymph_node_biopsy_breast"
    EXCISION_OF_BREAST_LUMP = "excision_of_breast_lump"
    CORE_BIOPSY_OF_BREAST = "core_biopsy_of_breast"
    INSERTION_OF_BREAST_IMPLANT = "insertion_of_breast_implant"
    BREAST_RECONSTRUCTION = "breast_reconstruction"
    INCISION_AND_DRAINAGE_OF_BREAST_ABSCESS = "incision_and_drainage_of_breast_abscess"
    MICRODOCHECTOMY = "microdochectomy"

    # C Eye
    PHACOEMULSIFICATION_CATARACT_EXTRACTION = "phacoemulsification_cataract_extraction"
    EXTRACAPSULAR_CATARACT_EXTRACTION = "extracapsular_cataract_extraction"
    INSERTION_OF_INTRAOCULAR_LENS = "insertion_of_intraocular_lens"
    YAG_LASER_CAPSULOTOMY = "yag_laser_capsulotomy"
    PARS_PLANA_VITRECTOMY = "pars_plana_vitrectomy"
    REPAIR_OF_RETINAL_DETACHMENT = "repair_of_retinal_detachment"
    SCLERAL_BUCKLE_PROCEDURE = "scleral_buckle_procedure"
    PNEUMATIC_RETINOPEXY = "pneumatic_retinopexy"
    LASER_PHOTOCOAGULATION_OF_RETINA = "laser_photocoagulation_of_retina"
    INTRAVITREAL_INJECTION = "intravitreal_injection"
    TRABECULECTOMY = "trabeculectomy"
    INSERTION_OF_GLAUCOMA_DRAINAGE_DEVICE = "insertion_of_glaucoma_drainage_device"
    CORNEAL_TRANSPLANT = "corneal_transplant"
    CORNEAL_GRAFT_LAMELLAR = "corneal_graft_lamellar"
    EXCISION_OF_PTERYGIUM = "excision_of_pterygium"
    STRABISMUS_CORRECTION_SURGERY = "strabismus_correction_surgery"
    EVISCERATION_OF_EYE = "evisceration_of_eye"
    ENUCLEATION_OF_EYE = "enucleation_of_eye"
    REPAIR_OF_PTOSIS = "repair_of_ptosis"
    CORRECTION_OF_ENTROPION_OR_ECTROPION = "correction_of_entropion_or_ectropion"

    # D Ear
    MYRINGOTOMY_WITH_GROMMET_INSERTION = "myringotomy_with_grommet_insertion"
    REMOVAL_OF_GROMMET = "removal_of_grommet"
    MYRINGOPLASTY = "myringoplasty"
    TYMPANOPLASTY = "tympanoplasty"
    MASTOIDECTOMY = "mastoidectomy"
    COCHLEAR_IMPLANT_INSERTION = "cochlear_implant_insertion"
    BONE_ANCHORED_HEARING_AID_INSERTION = "bone_anchored_hearing_aid_insertion"
    STAPEDECTOMY = "stapedectomy"
    EXCISION_OF_EXTERNAL_EAR_LESION = "excision_of_external_ear_lesion"
    MICROSUCTION_OF_EAR = "microsuction_of_ear"
    PINNAPLASTY = "pinnaplasty"

    # E Respiratory Tract
    LOBECTOMY_OF_LUNG = "lobectomy_of_lung"
    PNEUMONECTOMY = "pneumonectomy"
    SEGMENTECTOMY_OF_LUNG = "segmentectomy_of_lung"
    WEDGE_RESECTION_OF_LUNG = "wedge_resection_of_lung"
    VIDEO_ASSISTED_THORACOSCOPIC_SURGERY = "video_assisted_thoracoscopic_surgery"
    THORACOTOMY = "thoracotomy"
    PLEURODESIS = "pleurodesis"
    PLEURECTOMY = "pleurectomy"
    DECORTICATION_OF_LUNG = "decortication_of_lung"
    INSERTION_OF_CHEST_DRAIN = "insertion_of_chest_drain"
    TRACHEOSTOMY = "tracheostomy"
    PERCUTANEOUS_TRACHEOSTOMY = "percutaneous_tracheostomy"
    RIGID_BRONCHOSCOPY = "rigid_bronchoscopy"
    FLEXIBLE_BRONCHOSCOPY = "flexible_bronchoscopy"
    ENDOBRONCHIAL_STENT_INSERTION = "endobronchial_stent_insertion"
    MEDIASTINOSCOPY = "mediastinoscopy"
    LARYNGECTOMY = "laryngectomy"
    SEPTOPLASTY = "septoplasty"
    FUNCTIONAL_ENDOSCOPIC_SINUS_SURGERY = "functional_endoscopic_sinus_surgery"

    # F Mouth
    DENTAL_EXTRACTION = "dental_extraction"
    TONSILLECTOMY = "tonsillectomy"
    ADENOIDECTOMY = "adenoidectomy"
    ADENOTONSILLECTOMY = "adenotonsillectomy"
    REPAIR_OF_CLEFT_LIP = "repair_of_cleft_lip"
    REPAIR_OF_CLEFT_PALATE = "repair_of_cleft_palate"
    EXCISION_OF_ORAL_LESION = "excision_of_oral_lesion"
    GLOSSECTOMY = "glossectomy"
    EXCISION_OF_SALIVARY_GLAND = "excision_of_salivary_gland"
    PAROTIDECTOMY = "parotidectomy"
    DRAINAGE_OF_PERITONSILLAR_ABSCESS = "drainage_of_peritonsillar_abscess"
    FRENULOPLASTY = "frenuloplasty"

    # G Upper Digestive System
    OESOPHAGECTOMY = "oesophagectomy"
    IVOR_LEWIS_OESOPHAGECTOMY = "ivor_lewis_oesophagectomy"
    TRANSHIATAL_OESOPHAGECTOMY = "transhiatal_oesophagectomy"
    OESOPHAGOGASTRECTOMY = "oesophagogastrectomy"
    OESOPHAGEAL_STENT_INSERTION = "oesophageal_stent_insertion"
    OESOPHAGEAL_DILATATION = "oesophageal_dilatation"
    HELLER_MYOTOMY = "heller_myotomy"
    NISSEN_FUNDOPLICATION = "nissen_fundoplication"
    LAPAROSCOPIC_FUNDOPLICATION = "laparoscopic_fundoplication"
    HIATUS_HERNIA_REPAIR = "hiatus_hernia_repair"
    REPAIR_OF_PERFORATED_PEPTIC_ULCER = "repair_of_perforated_peptic_ulcer"
    OVERSEW_OF_BLEEDING_PEPTIC_ULCER = "oversew_of_bleeding_peptic_ulcer"
    TOTAL_GASTRECTOMY = "total_gastrectomy"
    SUBTOTAL_GASTRECTOMY = "subtotal_gastrectomy"
    PARTIAL_GASTRECTOMY = "partial_gastrectomy"
    DISTAL_GASTRECTOMY = "distal_gastrectomy"
    GASTROJEJUNOSTOMY = "gastrojejunostomy"
    PYLOROPLASTY = "pyloroplasty"
    PYLOROMYOTOMY = "pyloromyotomy"
    WEDGE_RESECTION_OF_STOMACH = "wedge_resection_of_stomach"
    OESOPHAGOGASTRODUODENOSCOPY = "oesophagogastroduodenoscopy"
    UPPER_GI_ENDOSCOPY_WITH_BIOPSY = "upper_gi_endoscopy_with_biopsy"
    ENDOSCOPIC_HAEMOSTASIS_UPPER_GI_BLEED = "endoscopic_haemostasis_upper_gi_bleed"
    ENDOSCOPIC_VARICEAL_BAND_LIGATION = "endoscopic_variceal_band_ligation"
    ENDOSCOPIC_SUBMUCOSAL_DISSECTION = "endoscopic_submucosal_dissection"
    ENDOSCOPIC_MUCOSAL_RESECTION = "endoscopic_mucosal_resection"
    LAPAROSCOPIC_SLEEVE_GASTRECTOMY = "laparoscopic_sleeve_gastrectomy"
    LAPAROSCOPIC_ROUX_EN_Y_GASTRIC_BYPASS = "laparoscopic_roux_en_y_gastric_bypass"
    LAPAROSCOPIC_ADJUSTABLE_GASTRIC_BANDING = "laparoscopic_adjustable_gastric_banding"
    REMOVAL_OF_GASTRIC_BAND = "removal_of_gastric_band"
    INSERTION_OF_GASTROSTOMY_TUBE = "insertion_of_gastrostomy_tube"
    PERCUTANEOUS_ENDOSCOPIC_GASTROSTOMY = "percutaneous_endoscopic_gastrostomy"
    DUODENAL_STENT_INSERTION = "duodenal_stent_insertion"
    OVERSEW_OF_DUODENAL_PERFORATION = "oversew_of_duodenal_perforation"
    PANCREATICODUODENECTOMY = "pancreaticoduodenectomy"

    # H Lower Digestive System
    RIGHT_HEMICOLECTOMY = "right_hemicolectomy"
    EXTENDED_RIGHT_HEMICOLECTOMY = "extended_right_hemicolectomy"
    LEFT_HEMICOLECTOMY = "left_hemicolectomy"
    SIGMOID_COLECTOMY = "sigmoid_colectomy"
    TOTAL_COLECTOMY = "total_colectomy"
    SUBTOTAL_COLECTOMY = "subtotal_colectomy"
    PANPROCTOCOLECTOMY = "panproctocolectomy"
    HARTMANNS_PROCEDURE = "hartmanns_procedure"
    REVERSAL_OF_HARTMANNS_PROCEDURE = "reversal_of_hartmanns_procedure"
    ANTERIOR_RESECTION_OF_RECTUM = "anterior_resection_of_rectum"
    ABDOMINOPERINEAL_RESECTION_OF_RECTUM = "abdominoperineal_resection_of_rectum"
    TOTAL_MESORECTAL_EXCISION = "total_mesorectal_excision"
    ILEOCAECAL_RESECTION = "ileocaecal_resection"
    SMALL_BOWEL_RESECTION = "small_bowel_resection"
    FORMATION_OF_LOOP_ILEOSTOMY = "formation_of_loop_ileostomy"
    FORMATION_OF_END_ILEOSTOMY = "formation_of_end_ileostomy"
    FORMATION_OF_LOOP_COLOSTOMY = "formation_of_loop_colostomy"
    FORMATION_OF_END_COLOSTOMY = "formation_of_end_colostomy"
    CLOSURE_OF_ILEOSTOMY = "closure_of_ileostomy"
    CLOSURE_OF_COLOSTOMY = "closure_of_colostomy"
    ILEOANAL_POUCH_FORMATION = "ileoanal_pouch_formation"
    APPENDICECTOMY = "appendicectomy"
    LAPAROSCOPIC_APPENDICECTOMY = "laparoscopic_appendicectomy"
    ADHESIOLYSIS = "adhesiolysis"
    LAPAROSCOPIC_ADHESIOLYSIS = "laparoscopic_adhesiolysis"
    SMALL_BOWEL_STRICTUROPLASTY = "small_bowel_stricturoplasty"
    COLONOSCOPY = "colonoscopy"
    FLEXIBLE_SIGMOIDOSCOPY = "flexible_sigmoidoscopy"
    ENDOSCOPIC_POLYPECTOMY = "endoscopic_polypectomy"
    COLONIC_STENT_INSERTION = "colonic_stent_insertion"
    HAEMORRHOIDECTOMY = "haemorrhoidectomy"
    STAPLED_HAEMORRHOIDOPEXY = "stapled_haemorrhoidopexy"
    RUBBER_BAND_LIGATION_OF_HAEMORRHOIDS = "rubber_band_ligation_of_haemorrhoids"
    LATERAL_SPHINCTEROTOMY = "lateral_sphincterotomy"
    EXAMINATION_UNDER_ANAESTHESIA_OF_ANUS = "examination_under_anaesthesia_of_anus"
    DRAINAGE_OF_PERIANAL_ABSCESS = "drainage_of_perianal_abscess"
    FISTULOTOMY = "fistulotomy"
    INSERTION_OF_SETON = "insertion_of_seton"
    EXCISION_OF_PILONIDAL_SINUS = "excision_of_pilonidal_sinus"
    OPEN_INGUINAL_HERNIA_REPAIR = "open_inguinal_hernia_repair"
    LAPAROSCOPIC_INGUINAL_HERNIA_REPAIR = "laparoscopic_inguinal_hernia_repair"
    TOTALLY_EXTRAPERITONEAL_HERNIA_REPAIR = "totally_extraperitoneal_hernia_repair"
    TRANSABDOMINAL_PREPERITONEAL_HERNIA_REPAIR = (
        "transabdominal_preperitoneal_hernia_repair"
    )
    FEMORAL_HERNIA_REPAIR = "femoral_hernia_repair"
    UMBILICAL_HERNIA_REPAIR = "umbilical_hernia_repair"
    INCISIONAL_HERNIA_REPAIR = "incisional_hernia_repair"
    VENTRAL_HERNIA_REPAIR = "ventral_hernia_repair"
    EPIGASTRIC_HERNIA_REPAIR = "epigastric_hernia_repair"
    PARASTOMAL_HERNIA_REPAIR = "parastomal_hernia_repair"

    # J Other Abdominal Organs, Principally Digestive
    LAPAROSCOPIC_CHOLECYSTECTOMY = "laparoscopic_cholecystectomy"
    OPEN_CHOLECYSTECTOMY = "open_cholecystectomy"
    SUBTOTAL_CHOLECYSTECTOMY = "subtotal_cholecystectomy"
    CHOLECYSTOSTOMY = "cholecystostomy"
    BILE_DUCT_EXPLORATION = "bile_duct_exploration"
    CHOLEDOCHOTOMY = "choledochotomy"
    HEPATICOJEJUNOSTOMY = "hepaticojejunostomy"
    ENDOSCOPIC_RETROGRADE_CHOLANGIOPANCREATOGRAPHY = (
        "endoscopic_retrograde_cholangiopancreatography"
    )
    ERCP_WITH_SPHINCTEROTOMY = "ercp_with_sphincterotomy"
    ERCP_WITH_STENT_INSERTION = "ercp_with_stent_insertion"
    ERCP_WITH_STONE_EXTRACTION = "ercp_with_stone_extraction"
    PERCUTANEOUS_TRANSHEPATIC_CHOLANGIOGRAM = "percutaneous_transhepatic_cholangiogram"
    LIVER_RESECTION = "liver_resection"
    LEFT_HEPATECTOMY = "left_hepatectomy"
    RIGHT_HEPATECTOMY = "right_hepatectomy"
    LIVER_SEGMENTECTOMY = "liver_segmentectomy"
    LIVER_WEDGE_RESECTION = "liver_wedge_resection"
    RADIOFREQUENCY_ABLATION_OF_LIVER_LESION = "radiofrequency_ablation_of_liver_lesion"
    LIVER_TRANSPLANTATION = "liver_transplantation"
    DRAINAGE_OF_LIVER_ABSCESS = "drainage_of_liver_abscess"
    LIVER_BIOPSY = "liver_biopsy"
    DISTAL_PANCREATECTOMY = "distal_pancreatectomy"
    TOTAL_PANCREATECTOMY = "total_pancreatectomy"
    WHIPPLES_PROCEDURE = "whipples_procedure"
    PANCREATIC_NECROSECTOMY = "pancreatic_necrosectomy"
    DRAINAGE_OF_PANCREATIC_PSEUDOCYST = "drainage_of_pancreatic_pseudocyst"
    SPLENECTOMY = "splenectomy"
    LAPAROSCOPIC_SPLENECTOMY = "laparoscopic_splenectomy"
    PARTIAL_SPLENECTOMY = "partial_splenectomy"

    # K Heart
    CORONARY_ARTERY_BYPASS_GRAFT = "coronary_artery_bypass_graft"
    OFF_PUMP_CORONARY_ARTERY_BYPASS = "off_pump_coronary_artery_bypass"
    PERCUTANEOUS_CORONARY_INTERVENTION = "percutaneous_coronary_intervention"
    CORONARY_ANGIOPLASTY_WITH_STENT = "coronary_angioplasty_with_stent"
    CORONARY_ANGIOGRAPHY = "coronary_angiography"
    AORTIC_VALVE_REPLACEMENT = "aortic_valve_replacement"
    TRANSCATHETER_AORTIC_VALVE_IMPLANTATION = "transcatheter_aortic_valve_implantation"
    MITRAL_VALVE_REPLACEMENT = "mitral_valve_replacement"
    MITRAL_VALVE_REPAIR = "mitral_valve_repair"
    TRICUSPID_VALVE_REPAIR = "tricuspid_valve_repair"
    TRICUSPID_VALVE_REPLACEMENT = "tricuspid_valve_replacement"
    PULMONARY_VALVE_REPLACEMENT = "pulmonary_valve_replacement"
    DOUBLE_VALVE_REPLACEMENT = "double_valve_replacement"
    PERMANENT_PACEMAKER_INSERTION = "permanent_pacemaker_insertion"
    PACEMAKER_BOX_CHANGE = "pacemaker_box_change"
    IMPLANTABLE_CARDIOVERTER_DEFIBRILLATOR_INSERTION = (
        "implantable_cardioverter_defibrillator_insertion"
    )
    CARDIAC_RESYNCHRONISATION_THERAPY_DEVICE_INSERTION = (
        "cardiac_resynchronisation_therapy_device_insertion"
    )
    LEAD_EXTRACTION = "lead_extraction"
    ELECTROPHYSIOLOGICAL_STUDY = "electrophysiological_study"
    CATHETER_ABLATION_FOR_ARRHYTHMIA = "catheter_ablation_for_arrhythmia"
    PULMONARY_VEIN_ISOLATION = "pulmonary_vein_isolation"
    DC_CARDIOVERSION = "dc_cardioversion"
    MAZE_PROCEDURE = "maze_procedure"
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
    ATRIAL_SEPTAL_DEFECT_REPAIR = "atrial_septal_defect_repair"
    VENTRICULAR_SEPTAL_DEFECT_REPAIR = "ventricular_septal_defect_repair"
    SURGICAL_CLOSURE_OF_PATENT_DUCTUS_ARTERIOSUS = (
        "surgical_closure_of_patent_ductus_arteriosus"
    )

    # L Arteries and Veins
    OPEN_ABDOMINAL_AORTIC_ANEURYSM_REPAIR = "open_abdominal_aortic_aneurysm_repair"
    ENDOVASCULAR_ANEURYSM_REPAIR = "endovascular_aneurysm_repair"
    THORACIC_ENDOVASCULAR_AORTIC_REPAIR = "thoracic_endovascular_aortic_repair"
    AORTIC_DISSECTION_REPAIR = "aortic_dissection_repair"
    CAROTID_ENDARTERECTOMY = "carotid_endarterectomy"
    CAROTID_ARTERY_STENTING = "carotid_artery_stenting"
    FEMOROPOPLITEAL_BYPASS = "femoropopliteal_bypass"
    FEMOROTIBIAL_BYPASS = "femorotibial_bypass"
    FEMOROFEMORAL_CROSSOVER_BYPASS = "femorofemoral_crossover_bypass"
    AXILLOFEMORAL_BYPASS = "axillofemoral_bypass"
    AORTOBIFEMORAL_BYPASS = "aortobifemoral_bypass"
    AORTOFEMORAL_BYPASS = "aortofemoral_bypass"
    LOWER_LIMB_ANGIOPLASTY = "lower_limb_angioplasty"
    LOWER_LIMB_ANGIOPLASTY_WITH_STENTING = "lower_limb_angioplasty_with_stenting"
    SURGICAL_EMBOLECTOMY = "surgical_embolectomy"
    THROMBECTOMY_OF_ARTERY = "thrombectomy_of_artery"
    CREATION_OF_ARTERIOVENOUS_FISTULA = "creation_of_arteriovenous_fistula"
    INSERTION_OF_ARTERIOVENOUS_GRAFT = "insertion_of_arteriovenous_graft"
    VARICOSE_VEIN_STRIPPING = "varicose_vein_stripping"
    ENDOVENOUS_LASER_ABLATION_OF_VARICOSE_VEINS = (
        "endovenous_laser_ablation_of_varicose_veins"
    )
    RADIOFREQUENCY_ABLATION_OF_VARICOSE_VEINS = (
        "radiofrequency_ablation_of_varicose_veins"
    )
    FOAM_SCLEROTHERAPY_OF_VARICOSE_VEINS = "foam_sclerotherapy_of_varicose_veins"
    PHLEBECTOMY = "phlebectomy"
    LIGATION_OF_VARICOSE_VEINS = "ligation_of_varicose_veins"
    INSERTION_OF_INFERIOR_VENA_CAVA_FILTER = "insertion_of_inferior_vena_cava_filter"
    VENOUS_THROMBECTOMY = "venous_thrombectomy"

    # M Urinary
    RADICAL_NEPHRECTOMY = "radical_nephrectomy"
    PARTIAL_NEPHRECTOMY = "partial_nephrectomy"
    SIMPLE_NEPHRECTOMY = "simple_nephrectomy"
    LAPAROSCOPIC_NEPHRECTOMY = "laparoscopic_nephrectomy"
    NEPHROURETERECTOMY = "nephroureterectomy"
    LIVE_DONOR_NEPHRECTOMY = "live_donor_nephrectomy"
    RENAL_TRANSPLANTATION = "renal_transplantation"
    PYELOPLASTY = "pyeloplasty"
    NEPHROSTOMY_INSERTION = "nephrostomy_insertion"
    PERCUTANEOUS_NEPHROLITHOTOMY = "percutaneous_nephrolithotomy"
    EXTRACORPOREAL_SHOCK_WAVE_LITHOTRIPSY = "extracorporeal_shock_wave_lithotripsy"
    URETEROSCOPY = "ureteroscopy"
    URETEROSCOPIC_LASER_LITHOTRIPSY = "ureteroscopic_laser_lithotripsy"
    INSERTION_OF_URETERIC_STENT = "insertion_of_ureteric_stent"
    REMOVAL_OF_URETERIC_STENT = "removal_of_ureteric_stent"
    URETERIC_REIMPLANTATION = "ureteric_reimplantation"
    RADICAL_CYSTECTOMY = "radical_cystectomy"
    PARTIAL_CYSTECTOMY = "partial_cystectomy"
    ILEAL_CONDUIT_FORMATION = "ileal_conduit_formation"
    TRANSURETHRAL_RESECTION_OF_BLADDER_TUMOUR = (
        "transurethral_resection_of_bladder_tumour"
    )
    CYSTOSCOPY = "cystoscopy"
    CYSTODIATHERMY = "cystodiathermy"
    INSERTION_OF_SUPRAPUBIC_CATHETER = "insertion_of_suprapubic_catheter"
    TRANSURETHRAL_RESECTION_OF_PROSTATE = "transurethral_resection_of_prostate"
    LASER_ENUCLEATION_OF_PROSTATE = "laser_enucleation_of_prostate"
    RADICAL_PROSTATECTOMY = "radical_prostatectomy"
    ROBOTIC_ASSISTED_RADICAL_PROSTATECTOMY = "robotic_assisted_radical_prostatectomy"
    TRANSRECTAL_ULTRASOUND_GUIDED_PROSTATE_BIOPSY = (
        "transrectal_ultrasound_guided_prostate_biopsy"
    )
    TRANSPERINEAL_PROSTATE_BIOPSY = "transperineal_prostate_biopsy"
    URETHROPLASTY = "urethroplasty"
    OPTICAL_URETHROTOMY = "optical_urethrotomy"
    URETHRAL_DILATATION = "urethral_dilatation"
    INSERTION_OF_ARTIFICIAL_URINARY_SPHINCTER = (
        "insertion_of_artificial_urinary_sphincter"
    )
    MID_URETHRAL_SLING_PROCEDURE = "mid_urethral_sling_procedure"
    COLPOSUSPENSION = "colposuspension"

    # N Male Genital Organs
    ORCHIDECTOMY = "orchidectomy"
    BILATERAL_ORCHIDECTOMY = "bilateral_orchidectomy"
    ORCHIDOPEXY = "orchidopexy"
    ORCHIDECTOMY_FOR_TESTICULAR_TORSION = "orchidectomy_for_testicular_torsion"
    DETORSION_OF_TESTIS = "detorsion_of_testis"
    VASECTOMY = "vasectomy"
    VASECTOMY_REVERSAL = "vasectomy_reversal"
    HYDROCELE_REPAIR = "hydrocele_repair"
    EPIDIDYMAL_CYST_EXCISION = "epididymal_cyst_excision"
    VARICOCELECTOMY = "varicocelectomy"
    CIRCUMCISION = "circumcision"
    PENILE_PROSTHESIS_INSERTION = "penile_prosthesis_insertion"
    NESBIT_PROCEDURE_FOR_PEYRONIES_DISEASE = "nesbit_procedure_for_peyronies_disease"
    DRAINAGE_OF_SCROTAL_ABSCESS = "drainage_of_scrotal_abscess"

    # P/Q Female Genital Tract
    TOTAL_ABDOMINAL_HYSTERECTOMY = "total_abdominal_hysterectomy"
    TOTAL_LAPAROSCOPIC_HYSTERECTOMY = "total_laparoscopic_hysterectomy"
    VAGINAL_HYSTERECTOMY = "vaginal_hysterectomy"
    SUBTOTAL_HYSTERECTOMY = "subtotal_hysterectomy"
    RADICAL_HYSTERECTOMY = "radical_hysterectomy"
    BILATERAL_SALPINGO_OOPHORECTOMY = "bilateral_salpingo_oophorectomy"
    UNILATERAL_SALPINGO_OOPHORECTOMY = "unilateral_salpingo_oophorectomy"
    OVARIAN_CYSTECTOMY = "ovarian_cystectomy"
    LAPAROSCOPIC_OVARIAN_CYSTECTOMY = "laparoscopic_ovarian_cystectomy"
    SALPINGECTOMY = "salpingectomy"
    SALPINGOSTOMY_FOR_ECTOPIC_PREGNANCY = "salpingostomy_for_ectopic_pregnancy"
    MYOMECTOMY = "myomectomy"
    LAPAROSCOPIC_MYOMECTOMY = "laparoscopic_myomectomy"
    ENDOMETRIAL_ABLATION = "endometrial_ablation"
    DILATATION_AND_CURETTAGE = "dilatation_and_curettage"
    HYSTEROSCOPY = "hysteroscopy"
    HYSTEROSCOPIC_POLYPECTOMY = "hysteroscopic_polypectomy"
    LAPAROSCOPIC_STERILISATION = "laparoscopic_sterilisation"
    CONE_BIOPSY_OF_CERVIX = "cone_biopsy_of_cervix"
    LOOP_EXCISION_OF_TRANSFORMATION_ZONE = "loop_excision_of_transformation_zone"
    COLPOSCOPY = "colposcopy"
    VAGINAL_REPAIR_ANTERIOR = "vaginal_repair_anterior"
    VAGINAL_REPAIR_POSTERIOR = "vaginal_repair_posterior"
    SACROCOLPOPEXY = "sacrocolpopexy"
    VULVECTOMY = "vulvectomy"

    # R Female Genital Tract Associated with Pregnancy, Childbirth and Puerperium
    LOWER_SEGMENT_CAESAREAN_SECTION = "lower_segment_caesarean_section"
    CLASSICAL_CAESAREAN_SECTION = "classical_caesarean_section"
    CAESAREAN_HYSTERECTOMY = "caesarean_hysterectomy"
    FORCEPS_DELIVERY = "forceps_delivery"
    VENTOUSE_DELIVERY = "ventouse_delivery"
    NORMAL_VAGINAL_DELIVERY = "normal_vaginal_delivery"
    BREECH_DELIVERY = "breech_delivery"
    INDUCTION_OF_LABOUR = "induction_of_labour"
    EPISIOTOMY = "episiotomy"
    REPAIR_OF_OBSTETRIC_PERINEAL_TEAR = "repair_of_obstetric_perineal_tear"
    MANUAL_REMOVAL_OF_PLACENTA = "manual_removal_of_placenta"
    EVACUATION_OF_RETAINED_PRODUCTS_OF_CONCEPTION = (
        "evacuation_of_retained_products_of_conception"
    )
    CERVICAL_CERCLAGE = "cervical_cerclage"
    EXTERNAL_CEPHALIC_VERSION = "external_cephalic_version"
    FETAL_BLOOD_SAMPLING = "fetal_blood_sampling"

    # S Skin
    EXCISION_OF_SKIN_LESION = "excision_of_skin_lesion"
    WIDE_LOCAL_EXCISION_OF_SKIN_LESION = "wide_local_excision_of_skin_lesion"
    EXCISION_OF_MALIGNANT_MELANOMA = "excision_of_malignant_melanoma"
    SHAVE_EXCISION_OF_SKIN_LESION = "shave_excision_of_skin_lesion"
    CURETTAGE_AND_CAUTERY_OF_SKIN_LESION = "curettage_and_cautery_of_skin_lesion"
    SURGICAL_DEBRIDEMENT_OF_SKIN_AND_SUBCUTANEOUS_TISSUE = (
        "surgical_debridement_of_skin_and_subcutaneous_tissue"
    )
    SPLIT_SKIN_GRAFT = "split_skin_graft"
    FULL_THICKNESS_SKIN_GRAFT = "full_thickness_skin_graft"
    LOCAL_SKIN_FLAP_RECONSTRUCTION = "local_skin_flap_reconstruction"
    EXCISION_OF_PILONIDAL_ABSCESS = "excision_of_pilonidal_abscess"
    EXCISION_OF_LIPOMA = "excision_of_lipoma"
    EXCISION_OF_SEBACEOUS_CYST = "excision_of_sebaceous_cyst"
    NEGATIVE_PRESSURE_WOUND_THERAPY = "negative_pressure_wound_therapy"

    # T Soft Tissue
    FASCIOTOMY = "fasciotomy"
    PLANTAR_FASCIA_RELEASE = "plantar_fascia_release"
    DUPUYTRENS_FASCIECTOMY = "dupuytrens_fasciectomy"
    EXCISION_OF_GANGLION = "excision_of_ganglion"
    BURSECTOMY = "bursectomy"
    TENDON_TRANSFER = "tendon_transfer"
    EXCISION_OF_TENDON_LESION = "excision_of_tendon_lesion"
    PRIMARY_TENDON_REPAIR = "primary_tendon_repair"
    SECONDARY_TENDON_REPAIR = "secondary_tendon_repair"
    ACHILLES_TENDON_REPAIR = "achilles_tendon_repair"
    ROTATOR_CUFF_REPAIR_OPEN = "rotator_cuff_repair_open"
    TENOLYSIS = "tenolysis"
    TENDON_LENGTHENING = "tendon_lengthening"
    PERCUTANEOUS_ACHILLES_TENOTOMY = "percutaneous_achilles_tenotomy"
    TRIGGER_FINGER_RELEASE = "trigger_finger_release"
    EXCISION_OF_TENDON_SHEATH = "excision_of_tendon_sheath"
    MUSCLE_BIOPSY = "muscle_biopsy"
    MUSCLE_REPAIR = "muscle_repair"
    CARPAL_TUNNEL_DECOMPRESSION = "carpal_tunnel_decompression"
    CUBITAL_TUNNEL_DECOMPRESSION = "cubital_tunnel_decompression"
    PERIPHERAL_NERVE_REPAIR = "peripheral_nerve_repair"
    PERIPHERAL_NERVE_DECOMPRESSION_OTHER = "peripheral_nerve_decompression_other"
    BLOCK_DISSECTION_OF_LYMPH_NODES = "block_dissection_of_lymph_nodes"
    EXCISION_OR_BIOPSY_OF_LYMPH_NODE = "excision_or_biopsy_of_lymph_node"

    # U Diagnostic Imaging, Testing and Rehabilitation (interventional radiology)
    IMAGE_GUIDED_BIOPSY = "image_guided_biopsy"
    IMAGE_GUIDED_DRAINAGE_OF_COLLECTION = "image_guided_drainage_of_collection"
    IMAGE_GUIDED_TUMOUR_ABLATION = "image_guided_tumour_ablation"
    THERAPEUTIC_EMBOLISATION = "therapeutic_embolisation"
    IMAGE_GUIDED_NERVE_BLOCK = "image_guided_nerve_block"

    # V Bones and Joints of Skull and Spine
    CRANIOTOMY_BONE_FLAP = "craniotomy_bone_flap"
    CRANIOPLASTY = "cranioplasty"
    LE_FORT_OSTEOTOMY = "le_fort_osteotomy"
    MANDIBULAR_OSTEOTOMY = "mandibular_osteotomy"
    ORIF_MANDIBLE_FRACTURE = "orif_mandible_fracture"
    ORIF_ZYGOMATIC_MAXILLARY_FRACTURE = "orif_zygomatic_maxillary_fracture"
    TEMPOROMANDIBULAR_JOINT_REPLACEMENT = "temporomandibular_joint_replacement"
    TEMPOROMANDIBULAR_JOINT_ARTHROSCOPY = "temporomandibular_joint_arthroscopy"
    CERVICAL_LAMINECTOMY_DECOMPRESSION = "cervical_laminectomy_decompression"
    LUMBAR_LAMINECTOMY_DECOMPRESSION = "lumbar_laminectomy_decompression"
    REVISION_LUMBAR_DECOMPRESSION = "revision_lumbar_decompression"
    LUMBAR_MICRODISCECTOMY = "lumbar_microdiscectomy"
    CERVICAL_MICRODISCECTOMY = "cervical_microdiscectomy"
    REVISION_LUMBAR_DISCECTOMY = "revision_lumbar_discectomy"
    LUMBAR_INTERSPINOUS_SPACER_INSERTION = "lumbar_interspinous_spacer_insertion"
    CERVICAL_DISC_REPLACEMENT = "cervical_disc_replacement"
    LUMBAR_DISC_REPLACEMENT = "lumbar_disc_replacement"
    ANTERIOR_CERVICAL_DISCECTOMY_AND_FUSION = "anterior_cervical_discectomy_and_fusion"
    POSTERIOR_CERVICAL_FUSION = "posterior_cervical_fusion"
    POSTERIOR_LUMBAR_FUSION = "posterior_lumbar_fusion"
    TRANSFORAMINAL_LUMBAR_INTERBODY_FUSION = "transforaminal_lumbar_interbody_fusion"
    ANTERIOR_LUMBAR_INTERBODY_FUSION = "anterior_lumbar_interbody_fusion"
    POSTERIOR_LUMBAR_INTERBODY_FUSION = "posterior_lumbar_interbody_fusion"
    REVISION_SPINAL_FUSION = "revision_spinal_fusion"
    POSTERIOR_INSTRUMENTED_FUSION_SPINE = "posterior_instrumented_fusion_spine"
    INSTRUMENTED_CORRECTION_SPINAL_DEFORMITY = (
        "instrumented_correction_spinal_deformity"
    )
    SCOLIOSIS_CORRECTION_SURGERY = "scoliosis_correction_surgery"
    EXCISION_OF_SPINAL_LESION = "excision_of_spinal_lesion"
    VERTEBROPLASTY = "vertebroplasty"
    KYPHOPLASTY = "kyphoplasty"
    REDUCTION_AND_FIXATION_OF_SPINAL_FRACTURE = (
        "reduction_and_fixation_of_spinal_fracture"
    )
    SPINAL_CORD_STIMULATOR_INSERTION = "spinal_cord_stimulator_insertion"
    BIOPSY_OF_SPINE = "biopsy_of_spine"
    FACET_JOINT_DENERVATION = "facet_joint_denervation"
    FACET_JOINT_INJECTION = "facet_joint_injection"
    EXPLORATION_OF_SPINE = "exploration_of_spine"
    MANIPULATION_OF_SPINE_UNDER_ANAESTHESIA = "manipulation_of_spine_under_anaesthesia"
    SPINAL_FORAMINOPLASTY = "spinal_foraminoplasty"
    PERCUTANEOUS_DISC_DECOMPRESSION = "percutaneous_disc_decompression"

    # W Other Bones and Joints
    COMPLEX_RECONSTRUCTION_OF_HAND = "complex_reconstruction_of_hand"
    COMPLEX_RECONSTRUCTION_OF_FOOT = "complex_reconstruction_of_foot"
    PROSTHETIC_REPLACEMENT_OF_BONE_SEGMENT = "prosthetic_replacement_of_bone_segment"
    EXCISION_OF_BONE_TUMOUR = "excision_of_bone_tumour"
    CURETTAGE_OF_BONE_LESION = "curettage_of_bone_lesion"
    EXCISION_OF_ECTOPIC_BONE = "excision_of_ectopic_bone"
    OSTEOTOMY = "osteotomy"
    HIGH_TIBIAL_OSTEOTOMY = "high_tibial_osteotomy"
    FEMORAL_OSTEOTOMY = "femoral_osteotomy"
    PELVIC_OSTEOTOMY = "pelvic_osteotomy"
    CALCANEAL_OSTEOTOMY = "calcaneal_osteotomy"
    FIRST_METATARSAL_OSTEOTOMY = "first_metatarsal_osteotomy"
    CORRECTIVE_OSTEOTOMY_OF_LONG_BONE = "corrective_osteotomy_of_long_bone"
    BONE_GRAFTING = "bone_grafting"
    BONE_MARROW_ASPIRATION_OR_BIOPSY = "bone_marrow_aspiration_or_biopsy"
    DRAINAGE_OF_BONE_ABSCESS = "drainage_of_bone_abscess"
    OPEN_REDUCTION_INTERNAL_FIXATION_LONG_BONE_FRACTURE = (
        "open_reduction_internal_fixation_long_bone_fracture"
    )
    OPEN_REDUCTION_INTERNAL_FIXATION_INTRAARTICULAR_FRACTURE = (
        "open_reduction_internal_fixation_intraarticular_fracture"
    )
    CLOSED_REDUCTION_INTERNAL_FIXATION_FRACTURE = (
        "closed_reduction_internal_fixation_fracture"
    )
    CLOSED_REDUCTION_EXTERNAL_FIXATION_FRACTURE = (
        "closed_reduction_external_fixation_fracture"
    )
    CLOSED_REDUCTION_OF_FRACTURE = "closed_reduction_of_fracture"
    DYNAMIC_HIP_SCREW_FIXATION = "dynamic_hip_screw_fixation"
    INTRAMEDULLARY_NAILING_FEMUR = "intramedullary_nailing_femur"
    INTRAMEDULLARY_NAILING_TIBIA = "intramedullary_nailing_tibia"
    INTRAMEDULLARY_NAILING_HUMERUS = "intramedullary_nailing_humerus"
    CANNULATED_SCREW_FIXATION_NECK_OF_FEMUR = "cannulated_screw_fixation_neck_of_femur"
    PLATE_FIXATION_OF_FRACTURE = "plate_fixation_of_fracture"
    EXTERNAL_FIXATION_OF_FRACTURE = "external_fixation_of_fracture"
    SKELETAL_TRACTION = "skeletal_traction"
    REMOVAL_OF_METALWORK = "removal_of_metalwork"
    TOTAL_HIP_REPLACEMENT_CEMENTED = "total_hip_replacement_cemented"
    TOTAL_HIP_REPLACEMENT_UNCEMENTED = "total_hip_replacement_uncemented"
    TOTAL_HIP_REPLACEMENT_HYBRID = "total_hip_replacement_hybrid"
    REVISION_TOTAL_HIP_REPLACEMENT = "revision_total_hip_replacement"
    HIP_HEMIARTHROPLASTY_CEMENTED = "hip_hemiarthroplasty_cemented"
    HIP_HEMIARTHROPLASTY_UNCEMENTED = "hip_hemiarthroplasty_uncemented"
    HIP_RESURFACING = "hip_resurfacing"
    TOTAL_KNEE_REPLACEMENT_CEMENTED = "total_knee_replacement_cemented"
    TOTAL_KNEE_REPLACEMENT_UNCEMENTED = "total_knee_replacement_uncemented"
    REVISION_TOTAL_KNEE_REPLACEMENT = "revision_total_knee_replacement"
    UNICOMPARTMENTAL_KNEE_REPLACEMENT = "unicompartmental_knee_replacement"
    PATELLOFEMORAL_JOINT_REPLACEMENT = "patellofemoral_joint_replacement"
    TOTAL_SHOULDER_REPLACEMENT = "total_shoulder_replacement"
    REVERSE_TOTAL_SHOULDER_REPLACEMENT = "reverse_total_shoulder_replacement"
    SHOULDER_HEMIARTHROPLASTY = "shoulder_hemiarthroplasty"
    TOTAL_ELBOW_REPLACEMENT = "total_elbow_replacement"
    TOTAL_ANKLE_REPLACEMENT = "total_ankle_replacement"
    RADIAL_HEAD_REPLACEMENT = "radial_head_replacement"
    EXCISION_ARTHROPLASTY = "excision_arthroplasty"
    INTERPOSITION_ARTHROPLASTY = "interposition_arthroplasty"
    ANKLE_ARTHRODESIS = "ankle_arthrodesis"
    SUBTALAR_ARTHRODESIS = "subtalar_arthrodesis"
    TRIPLE_ARTHRODESIS = "triple_arthrodesis"
    FIRST_MTP_JOINT_FUSION = "first_mtp_joint_fusion"
    WRIST_ARTHRODESIS = "wrist_arthrodesis"
    SHOULDER_ARTHRODESIS = "shoulder_arthrodesis"
    FUSION_OF_TOE_JOINT = "fusion_of_toe_joint"
    OPEN_REDUCTION_OF_JOINT_DISLOCATION = "open_reduction_of_joint_dislocation"
    CLOSED_REDUCTION_OF_JOINT_DISLOCATION = "closed_reduction_of_joint_dislocation"
    SYNOVECTOMY_OPEN = "synovectomy_open"
    OPEN_MENISCECTOMY = "open_meniscectomy"
    LIGAMENT_RECONSTRUCTION = "ligament_reconstruction"
    ACL_RECONSTRUCTION = "acl_reconstruction"
    PCL_RECONSTRUCTION = "pcl_reconstruction"
    MULTI_LIGAMENT_KNEE_RECONSTRUCTION = "multi_ligament_knee_reconstruction"
    LATERAL_LIGAMENT_RECONSTRUCTION_ANKLE = "lateral_ligament_reconstruction_ankle"
    LIGAMENT_REPAIR = "ligament_repair"
    JOINT_STABILISATION_PROCEDURE = "joint_stabilisation_procedure"
    SHOULDER_STABILISATION_PROCEDURE = "shoulder_stabilisation_procedure"
    RELEASE_OF_JOINT_CONTRACTURE = "release_of_joint_contracture"
    ARTHROSCOPIC_WASHOUT_OF_JOINT = "arthroscopic_washout_of_joint"
    OPEN_WASHOUT_OF_JOINT = "open_washout_of_joint"
    KNEE_ARTHROSCOPY_MENISCECTOMY = "knee_arthroscopy_meniscectomy"
    KNEE_ARTHROSCOPY_MENISCAL_REPAIR = "knee_arthroscopy_meniscal_repair"
    KNEE_ARTHROSCOPY_CARTILAGE_PROCEDURE = "knee_arthroscopy_cartilage_procedure"
    DIAGNOSTIC_KNEE_ARTHROSCOPY = "diagnostic_knee_arthroscopy"
    HIP_ARTHROSCOPY = "hip_arthroscopy"
    SHOULDER_ARTHROSCOPY_SUBACROMIAL_DECOMPRESSION = (
        "shoulder_arthroscopy_subacromial_decompression"
    )
    SHOULDER_ARTHROSCOPY_ROTATOR_CUFF_REPAIR = (
        "shoulder_arthroscopy_rotator_cuff_repair"
    )
    SHOULDER_ARTHROSCOPY_LABRAL_REPAIR = "shoulder_arthroscopy_labral_repair"
    ANKLE_ARTHROSCOPY = "ankle_arthroscopy"
    WRIST_ARTHROSCOPY = "wrist_arthroscopy"
    ELBOW_ARTHROSCOPY = "elbow_arthroscopy"
    DIAGNOSTIC_JOINT_ARTHROSCOPY_OTHER = "diagnostic_joint_arthroscopy_other"
    JOINT_ASPIRATION = "joint_aspiration"
    JOINT_INJECTION = "joint_injection"
    MANIPULATION_UNDER_ANAESTHESIA_OF_JOINT = "manipulation_under_anaesthesia_of_joint"

    # X Miscellaneous Operations
    REPLANTATION_OF_LIMB = "replantation_of_limb"
    AMPUTATION_ABOVE_KNEE = "amputation_above_knee"
    AMPUTATION_BELOW_KNEE = "amputation_below_knee"
    AMPUTATION_THROUGH_KNEE = "amputation_through_knee"
    AMPUTATION_OF_ARM = "amputation_of_arm"
    AMPUTATION_OF_HAND = "amputation_of_hand"
    AMPUTATION_OF_FOOT = "amputation_of_foot"
    AMPUTATION_OF_TOE = "amputation_of_toe"
    AMPUTATION_OF_FINGER = "amputation_of_finger"
    REVISION_OF_AMPUTATION_STUMP = "revision_of_amputation_stump"
    CORRECTION_OF_CONGENITAL_LIMB_DEFORMITY = "correction_of_congenital_limb_deformity"
    APPLICATION_OF_PLASTER_CAST = "application_of_plaster_cast"
    APPLICATION_OF_EXTERNAL_SPLINT = "application_of_external_splint"

    # General / cross-specialty (not chapter-specific in OPCS-4)
    DIAGNOSTIC_LAPAROSCOPY = "diagnostic_laparoscopy"
    EXPLORATORY_LAPAROTOMY = "exploratory_laparotomy"
    WOUND_DEBRIDEMENT = "wound_debridement"
    INCISION_AND_DRAINAGE_OF_ABSCESS = "incision_and_drainage_of_abscess"
    INSERTION_OF_CENTRAL_VENOUS_CATHETER = "insertion_of_central_venous_catheter"
    REMOVAL_OF_CENTRAL_VENOUS_CATHETER = "removal_of_central_venous_catheter"
    INSERTION_OF_PERIPHERALLY_INSERTED_CENTRAL_CATHETER = (
        "insertion_of_peripherally_inserted_central_catheter"
    )
    REMOVAL_OF_FOREIGN_BODY = "removal_of_foreign_body"
    EXAMINATION_UNDER_ANAESTHESIA = "examination_under_anaesthesia"


# ENUMS - COMPLICATIONS


class ComplicationType(str, Enum):
    """Type of complication occurring during, or noted in immediate relation to, the procedure.
    Use OTHER with complication_desc for a complication not listed here."""

    OTHER = "other"

    # General intraoperative
    HAEMORRHAGE_MAJOR_BLOOD_LOSS = "haemorrhage_major_blood_loss"
    VASCULAR_INJURY = "vascular_injury"
    VISCERAL_ORGAN_INJURY = "visceral_organ_injury"
    NERVE_INJURY_GENERAL = "nerve_injury_general"
    ANAESTHETIC_COMPLICATION = "anaesthetic_complication"
    DIFFICULT_OR_FAILED_AIRWAY = "difficult_or_failed_airway"
    ANAPHYLAXIS_ALLERGIC_REACTION = "anaphylaxis_allergic_reaction"
    EQUIPMENT_INSTRUMENT_FAILURE = "equipment_instrument_failure"
    CONVERSION_TO_OPEN = "conversion_to_open"
    WRONG_SITE_OR_PROCEDURE = "wrong_site_or_procedure"
    RETAINED_SURGICAL_ITEM = "retained_surgical_item"
    CARDIAC_ARREST = "cardiac_arrest"
    INTRAOPERATIVE_DEATH = "intraoperative_death"
    MEDICATION_ADMINISTRATION_ERROR = "medication_administration_error"
    INTRAOPERATIVE_CONTAMINATION_BREACH_OF_STERILITY = (
        "intraoperative_contamination_breach_of_sterility"
    )

    # Orthopaedic / MSK - neurovascular
    SCIATIC_NERVE_INJURY = "sciatic_nerve_injury"
    COMMON_PERONEAL_NERVE_INJURY = "common_peroneal_nerve_injury"
    FEMORAL_NERVE_INJURY = "femoral_nerve_injury"
    OBTURATOR_NERVE_INJURY = "obturator_nerve_injury"
    RADIAL_NERVE_INJURY = "radial_nerve_injury"
    ULNAR_NERVE_INJURY = "ulnar_nerve_injury"
    MEDIAN_NERVE_INJURY = "median_nerve_injury"
    NAMED_VESSEL_INJURY = "named_vessel_injury"

    # Orthopaedic / MSK - implant-related
    PERIPROSTHETIC_FRACTURE = "periprosthetic_fracture"
    IMPLANT_MALPOSITION_OR_MALALIGNMENT = "implant_malposition_or_malalignment"
    LEG_LENGTH_DISCREPANCY = "leg_length_discrepancy"
    INTRAOPERATIVE_DISLOCATION_OR_INSTABILITY = (
        "intraoperative_dislocation_or_instability"
    )
    SCREW_OR_GUIDEWIRE_MALPOSITION = "screw_or_guidewire_malposition"
    IMPLANT_MALFUNCTION_OR_BREAKAGE = "implant_malfunction_or_breakage"

    # Orthopaedic / MSK - fracture-related
    IATROGENIC_FRACTURE = "iatrogenic_fracture"

    # Orthopaedic / MSK - systemic / anaesthetic
    BONE_CEMENT_IMPLANTATION_SYNDROME = "bone_cement_implantation_syndrome"
    FAT_EMBOLISM_SYNDROME = "fat_embolism_syndrome"
    TOURNIQUET_RELATED_COMPLICATION = "tourniquet_related_complication"
    EXCESSIVE_BLOOD_LOSS_REQUIRING_TRANSFUSION = (
        "excessive_blood_loss_requiring_transfusion"
    )

    # Orthopaedic / MSK - other
    COMPARTMENT_SYNDROME = "compartment_syndrome"
    TENDON_OR_LIGAMENT_RUPTURE_IATROGENIC = "tendon_or_ligament_rupture_iatrogenic"


# ENUMS - SUPPORTING


class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    MIDLINE = "midline"
    NOT_APPLICABLE = "not_applicable"


class ProcedureUrgency(str, Enum):
    ELECTIVE = "elective"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    NOT_STATED = "not_stated"


class AnaestheticType(str, Enum):
    GENERAL = "general"
    REGIONAL = "regional"
    LOCAL = "local"
    SEDATION = "sedation"
    COMBINED = "combined"
    NOT_STATED = "not_stated"


# BLOCKS


class Implant(BaseModel):
    """A device or implant used or inserted during the procedure."""

    implant_desc: str = Field(
        description="Direct extract naming the implant/device (e.g. 'Exeter V40 cemented stem', 'DePuy Pinnacle acetabular shell', 'size 5 mesh')"
    )
    device_type: str | None = Field(
        None,
        description="General category of device, in your own words (e.g. 'femoral stem', 'plate', 'mesh', 'screw')",
    )
    manufacturer: str | None = Field(
        None, description="Manufacturer of the implant, if stated"
    )
    size_or_specification: str | None = Field(
        None,
        description="Size, offset, or other specification of the implant, if stated",
    )
    serial_or_lot_number: str | None = Field(
        None, description="Serial or batch/lot number of the implant, if stated"
    )


class ProcedureComplication(BaseModel):
    """A complication occurring during, or noted in immediate relation to, a procedure."""

    complication_type: ComplicationType = Field(
        description="Type of complication. Use OTHER if not in enum."
    )
    complication_desc: str = Field(
        description="Direct extract or close paraphrase describing the complication as documented"
    )
    management: str | None = Field(
        None,
        description="Direct extract of how the complication was managed, if stated",
    )


class Procedure(BaseModel):
    """A single procedure performed during the operation."""

    procedure_type: ProcedureType = Field(
        description="Specific procedure performed. Use OTHER with procedure_desc if not in enum."
    )
    procedure_desc: str | None = Field(
        None,
        description="Direct extract naming the procedure as documented. Required when procedure_type is OTHER; optional elaboration otherwise (e.g. approach, specific technique)",
    )
    laterality: Laterality | None = Field(
        None, description="Laterality of this procedure"
    )
    implants: list[Implant] = Field(
        default_factory=list,
        description="Implants or devices used in this procedure; empty if none reported",
    )


class ProcedureMetadata(BaseModel):
    """Operative metadata reported for the case, where stated. None/empty where not documented -
    do not infer or estimate."""

    urgency: ProcedureUrgency | None = Field(
        None,
        description="Elective, urgent, or emergency, if stated or clearly inferable from context (e.g. 'trauma list')",
    )
    anaesthetic_type: AnaestheticType | None = Field(
        None, description="Type of anaesthetic used, if stated"
    )
    operative_time_minutes: int | None = Field(
        None, ge=0, description="Total operative/procedure time in minutes, if stated"
    )
    estimated_blood_loss_ml: int | None = Field(
        None, ge=0, description="Estimated blood loss in millilitres, if stated"
    )
    tourniquet_time_minutes: int | None = Field(
        None, ge=0, description="Tourniquet time in minutes, if stated"
    )
    asa_grade: int | None = Field(
        None, ge=1, le=5, description="ASA physical status grade (1-5), if stated"
    )
    surgeon_grade: str | None = Field(
        None,
        description="Grade of the most senior operating surgeon as documented (e.g. 'consultant', 'registrar'), with names/identifiers redacted",
    )


# FINAL MODEL


class OperationNote(BaseModel):
    is_operation_note: bool = Field(
        description="True only if the document is an operative note or operation record describing a surgical/interventional procedure performed on a patient"
    )
    indication: str | None = Field(
        None, description="Indication for the operation, in your own words"
    )
    procedures: list[Procedure] = Field(
        default_factory=list,
        description="All procedures performed, in the order documented; empty if not an operation note",
    )
    findings: str | None = Field(
        None, description="Intraoperative or procedural findings as documented"
    )
    complications: list[ProcedureComplication] = Field(
        default_factory=list,
        description="Complications occurring during, or noted in immediate relation to, the procedure; empty if none reported",
    )
    metadata: ProcedureMetadata | None = Field(
        None,
        description="Operative metadata reported for the case; None if nothing stated",
    )
    operation_summary: str | None = Field(
        None,
        description="Short free-text overall summary of the operation performed; None if not an operation note",
    )
