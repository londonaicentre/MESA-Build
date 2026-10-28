# Opnoteschema

Operation notes

## Structure

```text
📁 opnoteschema
├── examples/            # Training examples showing document input and structured output
├── schema.py            # Pydantic model for specifying expected output structure
├── prompt_builder.py    # Prompt builder for data generation and inference
├── prompt_datagen.txt   # Prompt template with example (for training data generation)
├── prompt_main.txt      # Prompt template without example (for inference/deployment)
└── py.typed             # Type checking marker
```

## Usage

```python
from opnoteschema.prompt_builder import PromptBuilder

# Initialize builder
builder = PromptBuilder()

# Build data generation prompt (with example)
datagen_prompt = builder.build_datagen_prompt()

# Build main/inference prompt (without example)
main_prompt = builder.build_main_prompt()
```

## Schema

![Schema overview](https://londonaicentre.github.io/MESA-Build/schemas/opnoteschema.png)

| Type | Values |
| ---- | ------ |
| PreviousOperationRelation | return_to_theatre_for_complication, planned_staged_procedure, reversal, revision, completion, re_excision, removal_of_implant, treatment_of_recurrence, contralateral_or_other_site, other |
| ProcedureType | other, decompressive_craniectomy, excision_of_brain_lesion, brain_biopsy, evacuation_of_intracranial_haematoma, repair_of_cerebral_aneurysm, ventricular_shunt_procedure, external_ventricular_drain_insertion, insertion_of_intracranial_pressure_monitor, endoscopic_third_ventriculostomy, neuromodulation_device_insertion, microvascular_decompression_of_cranial_nerve, repair_of_dura, peripheral_nerve_repair_or_graft, peripheral_nerve_decompression, sympathectomy, total_thyroidectomy, subtotal_thyroidectomy, thyroid_lobectomy, completion_thyroidectomy, parathyroidectomy, adrenalectomy, mastectomy, wide_local_excision_of_breast, excision_of_breast_lesion, breast_reconstruction, breast_augmentation, breast_reduction, microdochectomy, cataract_extraction, secondary_intraocular_lens_insertion, vitreoretinal_surgery, ophthalmic_laser_procedure, trabeculectomy, insertion_of_glaucoma_drainage_device, corneal_graft, strabismus_surgery, eyelid_surgery, enucleation_or_evisceration_of_eye, myringotomy_with_grommet_insertion, tympanoplasty, mastoidectomy, stapedectomy, cochlear_implant_insertion, bone_anchored_hearing_aid_insertion, pinnaplasty, septoplasty, functional_endoscopic_sinus_surgery, tonsillectomy, adenoidectomy, laryngoscopy, laryngectomy, tracheostomy, parotidectomy, excision_of_submandibular_gland, neck_dissection, dental_extraction, excision_of_oral_lesion, glossectomy, frenuloplasty, lobectomy_of_lung, pneumonectomy, segmentectomy_of_lung, wedge_resection_of_lung, pleurodesis, pleurectomy, decortication_of_lung, insertion_of_chest_drain, bronchoscopy, mediastinoscopy, oesophagectomy, oesophageal_myotomy, fundoplication, hiatus_hernia_repair, repair_of_stomach_or_duodenum, total_gastrectomy, partial_gastrectomy, gastrojejunostomy, pyloroplasty_or_pyloromyotomy, bariatric_surgery, gastrostomy_insertion, upper_gi_endoscopy, right_hemicolectomy, transverse_colectomy, left_hemicolectomy, sigmoid_colectomy, subtotal_or_total_colectomy, panproctocolectomy, hartmanns_procedure, reversal_of_hartmanns_procedure, anterior_resection_of_rectum, abdominoperineal_resection_of_rectum, transanal_excision_of_rectal_lesion, ileocaecal_resection, small_bowel_resection, stricturoplasty, formation_of_ileostomy, formation_of_colostomy, closure_of_stoma, ileoanal_pouch_formation, appendicectomy, adhesiolysis, lower_gi_endoscopy, haemorrhoid_procedure, lateral_sphincterotomy, fistulotomy, insertion_of_seton, inguinal_hernia_repair, femoral_hernia_repair, ventral_hernia_repair, cholecystectomy, cholecystostomy, bile_duct_exploration, hepaticojejunostomy, ercp, percutaneous_transhepatic_biliary_procedure, major_hepatectomy, minor_liver_resection, liver_transplantation, pancreaticoduodenectomy, distal_pancreatectomy, total_pancreatectomy, pancreatic_necrosectomy, splenectomy, coronary_artery_bypass_graft, coronary_angiography, percutaneous_coronary_intervention, aortic_valve_replacement, mitral_valve_repair_or_replacement, tricuspid_or_pulmonary_valve_repair_or_replacement, cardiac_device_procedure, cardiac_ablation_or_electrophysiology_study, repair_of_septal_defect, pericardial_procedure, cardiac_transplantation, mechanical_circulatory_support_insertion, aortic_repair, carotid_endarterectomy, arterial_bypass, angioplasty_or_stenting, embolectomy_or_thrombectomy, arteriovenous_access_formation, varicose_vein_procedure, insertion_of_inferior_vena_cava_filter, vascular_access_device_insertion, vascular_access_device_removal, radical_nephrectomy, partial_nephrectomy, simple_nephrectomy, nephroureterectomy, renal_transplantation, pyeloplasty, nephrostomy_insertion, percutaneous_nephrolithotomy, ureteroscopy, ureteric_stent_procedure, ureteric_reimplantation, radical_cystectomy, partial_cystectomy, urinary_diversion, transurethral_resection_of_bladder_tumour, cystoscopy, insertion_of_suprapubic_catheter, transurethral_prostate_procedure, radical_prostatectomy, prostate_biopsy, urethral_procedure, incontinence_procedure, orchidectomy, orchidopexy, scrotal_procedure, vasectomy, vasectomy_reversal, circumcision, penile_procedure, total_hysterectomy, subtotal_hysterectomy, radical_hysterectomy, salpingo_oophorectomy, ovarian_cystectomy, salpingectomy, salpingotomy, myomectomy, omentectomy, cytoreductive_surgery, pelvic_exenteration, hysteroscopy, endometrial_ablation, uterine_curettage_or_evacuation, tubal_sterilisation, excision_of_cervix, vaginal_wall_repair, sacrocolpopexy, vulvectomy, caesarean_section, instrumental_delivery, repair_of_obstetric_perineal_tear, manual_removal_of_placenta, cervical_cerclage, excision_of_skin_or_subcutaneous_lesion, curettage_and_cautery_of_skin_lesion, skin_graft, local_or_regional_flap, free_flap, excision_of_soft_tissue_lesion, wound_washout_or_debridement, negative_pressure_wound_therapy, fasciotomy, fasciectomy, soft_tissue_release, tendon_repair, tendon_transfer, tendon_lengthening_or_tenotomy, muscle_biopsy, muscle_repair, lymph_node_dissection, sentinel_lymph_node_biopsy, excision_biopsy_of_lymph_node, percutaneous_biopsy, image_guided_drainage_of_collection, tumour_ablation, therapeutic_embolisation, nerve_block_or_spinal_injection, cranioplasty, orthognathic_surgery, temporomandibular_joint_procedure, spinal_decompression, discectomy, spinal_fusion, disc_replacement, vertebral_augmentation, excision_of_spinal_lesion, open_reduction_internal_fixation, closed_reduction_internal_fixation, intramedullary_nailing, external_fixation, closed_reduction_of_fracture, skeletal_traction, removal_of_metalwork, total_hip_replacement, hip_hemiarthroplasty, hip_resurfacing, total_knee_replacement, partial_knee_replacement, shoulder_replacement, revision_arthroplasty, excision_or_interposition_arthroplasty, endoprosthetic_replacement, excision_or_curettage_of_bone_lesion, osteotomy, bone_grafting, arthrodesis, reduction_of_joint_dislocation, synovectomy, meniscal_procedure, cartilage_procedure, ligament_reconstruction_or_repair, shoulder_stabilisation, subacromial_decompression, joint_washout, joint_aspiration_or_injection, manipulation_under_anaesthesia_of_joint, major_limb_amputation, minor_amputation, revision_of_amputation_stump, replantation_of_limb_or_digit, organ_retrieval, peritoneal_lavage, laparostomy, delayed_closure_of_abdomen, diagnostic_or_exploratory_procedure, incision_and_drainage_of_abscess, removal_of_foreign_body |
| SurgicalApproach | laparotomy, thoracotomy_or_sternotomy, craniotomy_or_craniectomy, burr_hole, open_other, laparoscopic, thoracoscopic, arthroscopic, endoscopic, robotic, percutaneous, endovascular, converted_to_open, other |
| Laterality | left, right, bilateral, not_applicable |
| ComplicationType | other, haemorrhage, blood_products_required, vascular_injury, nerve_injury, bowel_injury_or_enterotomy, bladder_or_ureteric_injury, bile_duct_injury, visceral_organ_injury, dural_tear_or_csf_leak, pneumothorax, tendon_or_ligament_injury, iatrogenic_fracture, periprosthetic_fracture, spillage_of_contents, anaesthetic_complication, difficult_or_failed_airway, anaphylaxis_allergic_reaction, cardiovascular_collapse, respiratory_failure, cardiac_arrest, seizure, fat_embolism_syndrome, tourniquet_related_complication, compartment_syndrome, equipment_instrument_failure, wrong_site_or_procedure, retained_surgical_item, medication_administration_error, breach_of_sterility, implant_or_guidewire_malposition, implant_failure_loosening_or_breakage, dislocation_or_instability, leg_length_discrepancy, anastomotic_or_graft_thrombosis, flap_compromise_or_failure, anastomotic_leak, surgical_site_infection, postoperative_collection, wound_dehiscence, venous_thromboembolism |
| OperationOutcome | completed_as_planned, more_than_planned, less_than_planned, different_to_planned, abandoned, intraoperative_death |
| ProcedureUrgency | elective, emergency |
| NcepodCategory | immediate, urgent, expedited, elective |
| AnaestheticType | general, spinal, epidural, peripheral_nerve_block, local, sedation |

## License

This project uses a proprietary license issued by Guy's and St Thomas' NHS Foundation Trust, enabling free (non-commercial) use by NHS organisations. See LICENSE files for details.
