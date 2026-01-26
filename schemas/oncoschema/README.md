# Oncoschema

Schema package for oncology extraction from cancer clinical documents.

## Structure

```text
📁 oncoschema
├── examples/            # Training examples showing document input and structured output
├── schema.py            # Pydantic model for specifying expected output structure
├── prompt_builder.py    # Prompt builder for data generation and inference
├── prompt_datagen.txt   # Prompt template with example (for training data generation)
├── prompt_main.txt      # Prompt template without example (for inference/deployment)
└── py.typed             # Type checking marker
```

## Usage

```python
from oncoschema.prompt_builder import PromptBuilder

# Initialize builder
builder = PromptBuilder()

# Build data generation prompt (with example)
datagen_prompt = builder.build_datagen_prompt()

# Build main/inference prompt (without example)
main_prompt = builder.build_main_prompt()
```

## Schema

![Schema overview](https://londonaicentre.github.io/MESA-Build/schemas/oncoschema.png)

| Type | Values |
| ---- | ------ |
| TopographyType | unknown_primary, other, haematological, lung, pleura, other_respiratory, oesophagus, stomach, small_intestine, colon, rectum, pancreas, liver, gallbladder, bile_duct, other_gi, kidney, bladder, prostate, testis, other_gu, breast, cervix, uterus, ovary, other_gynae, brain, spinal_cord, other_cns, oral, hypo_oro_naso_pharynx, larynx, salivary_gland, nasal_cavity, paranasal_sinus, thyroid, adrenal_gland, other_endocrine, skin, soft_tissue, bone |
| MorphologyType | unknown_morphology, other, adenocarcinoma, squamous_cell_carcinoma, urothelial_carcinoma, renal_cell_carcinoma, hepatocellular_carcinoma, small_cell_carcinoma, non_small_cell_carcinoma, carcinoma_other, mesothelioma, melanoma, neuroendocrine, carcinoid, sarcoma_nos, gastrointestinal_stromal_tumour, osteosarcoma, chondrosarcoma, ewing_sarcoma, rhabdomyosarcoma, kaposi_sarcoma, soft_tissue_sarcoma, acute_lymphoblastic_leukaemia, acute_myeloid_leukaemia, chronic_lymphocytic_leukaemia, chronic_myeloid_leukaemia, hodgkin_lymphoma, non_hodgkin_lymphoma, multiple_myeloma, leukaemia_other, myelodysplastic, glioblastoma, astrocytoma, oligodendroglioma, meningioma, seminoma, teratoma, choriocarcinoma, wilms_tumour, retinoblastoma, hepatoblastoma |
| MSIStatus | msi_high, ms_stable |
| TMBStatus | tmb_high, tmb_low, tmb_intermediate |
| MolecularBiomarkerType | other, braf, ntrk1, ntrk2, ntrk3, ret, erbb2_her2, tp53, brca1, brca2, mlh1, msh2, msh6, pms2, palb2, rad51, egfr, alk, ros1, met, kras, nras, pik3ca, esr1_er, pgr_pr, ki_67, kit, pdgfra, fgfr1, fgfr2, fgfr3, idh1, idh2, pdl1, nf1, nf2, mgmt, npm1, flt3, bcr_abl1, jak2, bcl2, myc |
| BiomarkerStatus | altered, negative, equivocal, hypothetical |
| CancerScoreName | pathological_grade, gleason, figo, dukes, breslow, clark, binet, child_pugh, other |
| SpreadType | other, lymph_node, liver, lung, spine, other_bone, brain, other_cns, adrenal, kidney, pleura, peritoneum, skin, pancreas, spleen, ovary, testis, thyroid, stomach, bowel, bladder, prostate, breast, head_and_neck |
| TimelineEventType | had_systemic_or_radiotherapy_treatment, had_surgical_treatment_performed, experienced_toxicity_or_complication_related_to_treatment, experienced_treatment_reduction_or_stop, considered_for_clinical_trial, enrolled_to_clinical_trial, positive_treatment_response_on_assessment, evidence_of_metastatic_progression, radiology_evidence_of_disease_progression, experienced_disease_remission, experienced_disease_recurrence, patient_died |
| PatientFindingType | comorbidity_finding, social_or_family_finding, symptom_finding, physical_examination_finding, functional_finding, mental_state_finding |
| PatientFindingStatus | is_present, is_not_present, uncertain |
| FuturePlanType | planned_systemic_or_radiotherapy_treatment, planned_surgery_treatment, planned_investigation, planned_clinical_trial_involvement |

## License

This project uses a proprietary license issued by Guy's and St Thomas' NHS Foundation Trust, enabling free (non-commercial) use by NHS organisations. See LICENSE files for details.
