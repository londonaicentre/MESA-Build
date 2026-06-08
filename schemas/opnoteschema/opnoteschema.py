from pydantic import BaseModel, Field
from typing import Optional, List, Annotated
from enum import Enum

Year = Annotated[int, Field(ge=1900, le=2100)]

class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    MIDLINE = "midline"
    NOT_APPLICABLE = "not_applicable"

## Enums

# Procedures
class Procedure(str, Enum):
    # Hip
    TOTAL_HIP_REPLACEMENT = "Total Hip Replacement"
    HIP_HEMIARTHROPLASTY = "Hip Hemiarthroplasty"
    DYNAMIC_HIP_SCREW_FIXATION = "Dynamic Hip Screw Fixation"
    INTRAMEDULLARY_HIP_NAIL = "Intramedullary Hip Nail"
    HIP_ARTHROSCOPY = "Hip Arthroscopy"

    # Knee
    TOTAL_KNEE_REPLACEMENT = "Total Knee Replacement"
    UNICOMPARTMENTAL_KNEE_REPLACEMENT = "Unicompartmental Knee Replacement"
    ACL_RECONSTRUCTION = "ACL Reconstruction"
    KNEE_ARTHROSCOPY = "Knee Arthroscopy"
    TIBIAL_PLATEAU_ORIF = "Tibial Plateau ORIF"
    PATELLA_ORIF = "Patella ORIF"
    HIGH_TIBIAL_OSTEOTOMY = "High Tibial Osteotomy"

    # Shoulder & Elbow
    TOTAL_SHOULDER_REPLACEMENT = "Total Shoulder Replacement"
    SHOULDER_HEMIARTHROPLASTY = "Shoulder Hemiarthroplasty"
    ROTATOR_CUFF_REPAIR = "Rotator Cuff Repair"
    SUBACROMIAL_DECOMPRESSION = "Subacromial Decompression"
    SHOULDER_STABILISATION = "Shoulder Stabilisation"
    TOTAL_ELBOW_REPLACEMENT = "Total Elbow Replacement"
    DISTAL_HUMERUS_ORIF = "Distal Humerus ORIF"
    RADIAL_HEAD_ORIF = "Radial Head ORIF"
    RADIAL_HEAD_ARTHROPLASTY = "Radial Head Arthroplasty"

    # Foot & Ankle
    TOTAL_ANKLE_ARTHROPLASTY = "Total Ankle Arthroplasty"
    ANKLE_ARTHRODESIS = "Ankle Arthrodesis"
    HALLUX_VALGUS_CORRECTION = "Hallux Valgus Correction"
    FIRST_MTP_JOINT_FUSION = "First MTP Joint Fusion"
    ANKLE_FRACTURE_ORIF = "Ankle Fracture ORIF"
    CALCANEAL_ORIF = "Calcaneal ORIF"
    ACHILLES_TENDON_REPAIR = "Achilles Tendon Repair"
    SUBTALAR_ARTHRODESIS = "Subtalar Arthrodesis"
    TRIPLE_ARTHRODESIS = "Triple Arthrodesis"

    # Spine
    LUMBAR_MICRODISCECTOMY = "Lumbar Microdiscectomy"
    LUMBAR_DECOMPRESSION = "Lumbar Decompression"
    POSTERIOR_LUMBAR_FUSION = "Posterior Lumbar Fusion"
    ACDF = "Anterior Cervical Discectomy and Fusion"
    CERVICAL_DISC_ARTHROPLASTY = "Cervical Disc Arthroplasty"
    VERTEBROPLASTY = "Vertebroplasty"
    KYPHOPLASTY = "Kyphoplasty"
    POSTERIOR_INSTRUMENTED_FUSION = "Posterior Instrumented Fusion"

    # Hand & Wrist
    CARPAL_TUNNEL_DECOMPRESSION = "Carpal Tunnel Decompression"
    TRIGGER_FINGER_RELEASE = "Trigger Finger Release"
    DUPUYTRENS_FASCIECTOMY = "Dupuytren's Fasciectomy"
    DISTAL_RADIUS_ORIF = "Distal Radius ORIF"
    SCAPHOID_FIXATION = "Scaphoid Fixation"
    THUMB_CMC_ARTHROPLASTY = "Thumb CMC Arthroplasty"
    WRIST_ARTHRODESIS = "Wrist Arthrodesis"
    EXTENSOR_TENDON_REPAIR = "Extensor Tendon Repair"

    # Long Bone Trauma
    FEMORAL_SHAFT_INTRAMEDULLARY_NAIL = "Femoral Shaft Intramedullary Nail"
    TIBIAL_SHAFT_INTRAMEDULLARY_NAIL = "Tibial Shaft Intramedullary Nail"
    FEMORAL_SHAFT_ORIF = "Femoral Shaft ORIF"
    HUMERAL_SHAFT_INTRAMEDULLARY_NAIL = "Humeral Shaft Intramedullary Nail"
    EXTERNAL_FIXATOR_APPLICATION = "External Fixator Application"
    SUPRACONDYLAR_FEMUR_ORIF = "Supracondylar Femur ORIF"

    # Paediatric
    FEMORAL_OSTEOTOMY = "Femoral Osteotomy"
    PELVIC_OSTEOTOMY = "Pelvic Osteotomy"
    SUPRACONDYLAR_HUMERUS_FIXATION = "Supracondylar Humerus Fixation"
    FLEXIBLE_INTRAMEDULLARY_NAILING = "Flexible Intramedullary Nailing"
    PERCUTANEOUS_ACHILLES_TENOTOMY = "Percutaneous Achilles Tenotomy"
    SLIPPED_UPPER_FEMORAL_EPIPHYSIS_FIXATION = "Slipped Upper Femoral Epiphysis Fixation"
    SCOLIOSIS_POSTERIOR_INSTRUMENTED_FUSION = "Scoliosis Posterior Instrumented Fusion"

    UNKNOWN = "Unknown"
    OTHER = "Other"

class ProcedureModifier(str, Enum):
    PRIMARY = "Primary"
    REVISION = "Revision"
    UNKNOWN = "Unknown"


# Implant
class OrthoImplant(str, Enum):

    # Hip Arthroplasty
    PINNACLE_SECTOR_II_SHELL = "Pinnacle Sector II Shell"
    PINNACLE_ALTRX_PE_LINER = "Pinnacle ALTRX PE Liner"
    CORAIL_STANDARD_OFFSET_STEM = "Corail Standard Offset Stem"
    BIOLOX_DELTA_CERAMIC_HEAD = "Biolox Delta Ceramic Head"
    EXETER_V40_STEM = "Exeter V40 Stem"
    EXETER_CONTEMPORARY_ACETABULAR_CUP = "Exeter Contemporary Acetabular Cup"
    EXETER_X3_RIMFIT_PE_LINER = "Exeter X3 Rimfit PE Liner"
    POLARSTEM_COLLARED_STEM = "Polarstem Collared Stem"
    R3_ACETABULAR_SHELL = "R3 Acetabular Shell"
    R3_XLPE_LINER = "R3 XLPE Liner"
    TRINITY_ACETABULAR_CUP = "Trinity Acetabular Cup"
    TRINITY_DUAL_MOBILITY_LINER = "Trinity Dual Mobility Liner"
    RECAP_REVISION_ACETABULAR_CAGE = "ReCap Revision Acetabular Cage"
    FURLONG_HAC_STEM = "Furlong HAC Stem"

    # Knee Arthroplasty
    TRIATHLON_CR_TOTAL_KNEE = "Triathlon CR Total Knee"
    TRIATHLON_TIBIAL_BASEPLATE = "Triathlon Tibial Baseplate"
    TRIATHLON_X3_FIXED_BEARING_INSERT = "Triathlon X3 Fixed Bearing Insert"
    ATTUNE_CR_FEMORAL_COMPONENT = "Attune CR Femoral Component"
    ATTUNE_TIBIAL_BASE = "Attune Tibial Base"
    ATTUNE_ARTICULATION_INSERT = "Attune Articulation Insert"
    JOURNEY_II_CR_FEMORAL = "Journey II CR Femoral"
    JOURNEY_II_TIBIAL_BASEPLATE = "Journey II Tibial Baseplate"
    JOURNEY_II_INSERT = "Journey II Insert"
    PERSONA_CR_FEMORAL = "Persona CR Femoral"
    PERSONA_TIBIAL_BASE = "Persona Tibial Base"
    PERSONA_INSERT = "Persona Insert"
    OXFORD_UNICOMPARTMENTAL_FEMORAL = "Oxford Unicompartmental Femoral"
    OXFORD_UNICOMPARTMENTAL_TIBIAL = "Oxford Unicompartmental Tibial"
    OXFORD_MENISCAL_BEARING = "Oxford Meniscal Bearing"

    # Shoulder Arthroplasty
    DELTA_XTEND_HUMERAL_STEM = "Delta XTEND Humeral Stem"
    DELTA_XTEND_GLENOID = "Delta XTEND Glenoid"
    ECLIPSE_HUMERAL_STEM = "Eclipse Humeral Stem"
    ECLIPSE_GLENOID = "Eclipse Glenoid"
    COMPREHENSIVE_REVERSE_HUMERAL_STEM = "Comprehensive Reverse Humeral Stem"
    COMPREHENSIVE_BASEPLATE = "Comprehensive Baseplate"
    AFFINIS_SHORT_STEM = "Affinis Short Stem"

    # Trauma
    PHILOS_PLATE = "PHILOS Plate"
    LCP_DISTAL_FEMORAL_PLATE = "LCP Distal Femoral Plate"
    TFNA_TROCHANTERIC_FIXATION_NAIL_LONG = "TFNA (Trochanteric Fixation Nail Long)"
    INTERTAN_NAIL = "InterTAN Nail"
    META_NAIL_TIBIAL = "Meta-Nail Tibial"
    VARIAX_DISTAL_RADIUS_PLATE = "VariAx Distal Radius Plate"
    ACUTRAK_2_HEADLESS_SCREW = "Acutrak 2 Headless Screw"

    # Upper Limb
    HERBERT_WHIPPLE_SCREW = "Herbert Whipple Screw"
    RCP_RADIAL_HEAD = "RCP Radial Head"
    EVOLVE_RADIAL_HEAD = "Evolve Radial Head"
    LATITUDE_TOTAL_ELBOW = "Latitude Total Elbow"

    # Foot & Ankle
    INBONE_II_TOTAL_ANKLE = "INBONE II Total Ankle"
    INBONE_II_TIBIAL_COMPONENT = "INBONE II Tibial Component"
    INFINITY_TOTAL_ANKLE = "Infinity Total Ankle"
    STAR_TOTAL_ANKLE = "STAR Total Ankle"
    CADENCE_ANKLE_ARTHRODESIS_NAIL = "Cadence Ankle Arthrodesis Nail"

    # Spinal
    EXPEDIUM_PEDICLE_SCREW = "Expedium Pedicle Screw"
    XIA_PEDICLE_SCREW = "XIA Pedicle Screw"
    MESA_TLIF_CAGE = "Mesa TLIF Cage"

    UNKNOWN = "Unknown"
    OTHER = "Other"



# Findings - broad, does enum list make sense?
    

class Procedure(BaseModel):
    procedure_name: Procedure = Field(..., description="Name of procedure")
    laterality: Optional[Laterality] = Field(default=None, description="Laterality of the procedure")

class Implant(BaseModel):
    device_type: Optional[str] = Field(None, description="e.g. femoral stem, pacemaker lead, mesh, stent, IOL")
    implant_name: Optional[OrthoImplant] = Field(None, description="Commercial name of the implant")
    manufacturer: Optional[str] = Field(None, description="Manufacturer of the implant")
    serial_number: Optional[str] = Field(None, description="Serial number of the implant")

class OperationNote(BaseModel):
    year: Year
    indication: Optional[str] = Field(None, description="Indication for the operation")
    procedures: List[Procedure] = Field(default_factory=list, description="All procedures performed, in order documented")
    findings: Optional[str] = Field(None, description="Intraoperative or procedural findings as documented")
    implants: Optional[List[Implant]] = Field(None, description="All implants implanted, in order documented")