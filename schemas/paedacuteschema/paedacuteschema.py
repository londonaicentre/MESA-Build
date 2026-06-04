from pydantic import BaseModel, Field
from typing import List, Optional

class PaediatricSymptom(BaseModel):
    
    # BEHAVIOUR & INTERACTION
    social_interaction_absent: bool = Field(
        default=False,
        description="Child does not engage with parent/carer, ignores voice, no eye contact"
    )
    absent_social_smile: bool = Field(
        default=False,
        description="No smiling in response to parent/nurse (age >2 months)"
    )
    play_behaviour_reduced: bool = Field(
        default=False,
        description="Not interested in toys, games, or usual activities"
    )
    
    # CONSCIOUSNESS & ALERTNESS
    drowsy_hard_to_wake: bool = Field(
        default=False,
        description="Lethargic, sleepy, difficult to rouse, reduced response to stimuli"
    )
    floppy_limp_tone: bool = Field(
        default=False,
        description="Reduced muscle tone, feels floppy when held, poor head control"
    )
    confused_disoriented: bool = Field(
        default=False,
        description="Not making sense, disoriented to person/place/time, inappropriate speech"
    )
    
    # RESPIRATORY (specific signs)
    grunting: bool = Field(
        default=False,
        description="Expiratory grunting — audible on auscultation or at rest"
    )
    nasal_flaring: bool = Field(
        default=False,
        description="Nostrils flare with each breath"
    )
    recessions: bool = Field(
        default=False,
        description="Intercostal, subcostal, or sternal recession (chest indrawing)"
    )
    apnoea_or_pauses: bool = Field(
        default=False,
        description="Stops breathing, has pauses, or desaturation episodes"
    )
    wheeze_audible: bool = Field(
        default=False,
        description="Audible wheeze without stethoscope, or documented wheeze"
    )
    
    # FEEDING & HYDRATION 
    poor_feeding_reduced_intake: bool = Field(
        default=False,
        description="Taking <50% normal feeds, not interested in breast/bottle, reduced wet nappies"
    )
    unable_to_drink: bool = Field(
        default=False,
        description="Cannot swallow, dribbling, refusing all oral intake"
    )
    
    # SKIN & PERFUSION
    pallor: bool = Field(
        default=False,
        description="Pale appearance — documented as 'pale', 'pallor'"
    )
    mottled_skin: bool = Field(
        default=False,
        description="Mottling, marbled appearance, lace-like pattern"
    )
    cyanosis: bool = Field(
        default=False,
        description="Blue discolouration — lips, tongue, extremities, or central"
    )
    rash_blanching: bool = Field(
        default=False,
        description="Blanching rash — fades with pressure; includes viral exanthem, HHV-6, roseola"
    )
    rash_non_blanching: bool = Field(
        default=False,
        description="Non-blanching rash — petechiae or purpura that do not fade with pressure; suspected meningococcaemia until proven otherwise"
    )
    
    # MOTOR & NEUROLOGICAL
    seizure_activity: bool = Field(
        default=False,
        description="Any seizure, convulsion, fit — focal or generalised"
    )
    posturing: bool = Field(
        default=False,
        description="Decorticate/decerebrate posturing, opisthotonos"
    )
    not_able_to_stand_or_walk: bool = Field(
        default=False,
        description="Child previously mobile now cannot stand/sit/walk independently"
    )
    bulging_fontanelle: bool = Field(
        default=False,
        description="Anterior fontanelle bulging at rest (not during crying), relevant in infants with open fontanelle only"
    )

    
    # CRYING & VOCALISATION
    high_pitched_cry: bool = Field(
        default=False,
        description="Neurogenic or abnormally high-pitched cry"
    )
    weak_or_absent_cry: bool = Field(
        default=False,
        description="Feeble cry, unable to cry, crying without sound"
    )
    inconsolable_crying: bool = Field(
        default=False,
        description="Crying that does not stop with usual comforting measures"
    )
    
    # CARER CONCERN (highly predictive)
    parent_says_not_themselves: bool = Field(
        default=False,
        description="Parent explicitly states child is 'not themselves', 'different', or 'not right'"
    )
    parent_instinct_concerned: bool = Field(
        default=False,
        description="Parent expresses strong concern or 'I know my child' statement"
    )
    
    # HIGH-RISK CONTEXT
    immunocompromised: bool = Field(
        default=False,
        description="Known immunodeficiency, chemotherapy, asplenia, transplant, high-dose steroids"
    )
    recent_hospital_discharge: bool = Field(
        default=False,
        description="Discharged from hospital in last 48-72 hours"
    )
    no_fixed_abode_or_vulnerable: bool = Field(
        default=False,
        description="Homeless, looked after child, safeguarding concern"
    )
    
    # GENERAL APPEARANCE (clinician gestalt)
    clinician_concern_unwell: bool = Field(
        default=False,
        description="Clinician explicitly documents 'looks unwell', 'toxic', or concerning appearance"
    )