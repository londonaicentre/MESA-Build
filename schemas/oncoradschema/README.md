# Oncoradschema

Schema package for oncology extraction from radiology reports.

## Structure

```text
📁 oncoradschema
├── examples/            # Training examples showing document input and structured output
├── schema.py            # Pydantic model for specifying expected output structure
├── prompt_builder.py    # Prompt builder for data generation and inference
├── prompt_datagen.txt   # Prompt template with example (for training data generation)
├── prompt_main.txt      # Prompt template without example (for inference/deployment)
└── py.typed             # Type checking marker
```

## Usage

```python
from oncoradschema.prompt_builder import PromptBuilder

# Initialize builder
builder = PromptBuilder()

# Build data generation prompt (with example)
datagen_prompt = builder.build_datagen_prompt()

# Build main/inference prompt (without example)
main_prompt = builder.build_main_prompt()
```

## Schema

![Schema overview](https://londonaicentre.github.io/MESA-Build/schemas/oncoradschema.png)

| Type | Values |
| ---- | ------ |
| ScanModality | ct, mri, pet, pet_ct, pet_mri, spect, spect_ct, scintigraphy, ultrasound, xray, fluoroscopy, mammography, tomosynthesis, other |
| ScanContrast | without_contrast, with_contrast, dual_phase, triple_phase |
| ScanRegion | head, neck, chest, abdomen, pelvis, spine, upper_limb, lower_limb, whole_body, heart, breast, axilla, vessel, other |
| ScanRationale | routine_screening, diagnostic_pre_treatment, treatment_planning, follow_up, interventional, not_determinable |
| ComparativeChange | new, progressive, stable, improving, resolved, mixed, indeterminate |
| FormalRECISTResponse | complete_response, partial_response, stable_disease, progressive_disease, not_evaluable, not_applicable |
| ScoringSystem | other, tnm, ajcc_stage, rcr, bi_rads, acr_breast_density, pi_rads, bosniak, li_rads, lung_rads, ti_rads, o_rads, deauville, gleason |
| Laterality | left, right, bilateral, not_applicable |
| AnatomicalSite | other, unknown, lung, pleura, mediastinum, heart, chest_wall, liver, pancreas, spleen, gallbladder, bile_duct, kidney, adrenal, oesophagus, stomach, small_intestine, colon, rectum, anus, peritoneum, omentum, mesentery, retroperitoneum, bladder, prostate, testis, ovary, uterus, cervix, breast, brain, meninges, spinal_cord, oral_cavity, pharynx, larynx, salivary_gland, nasal_cavity, paranasal_sinus, thyroid, bone_skull, bone_spine, bone_ribs_sternum, bone_pelvis, bone_upper_limb, bone_lower_limb, soft_tissue_head_neck, soft_tissue_thorax, soft_tissue_abdomen, soft_tissue_pelvis, soft_tissue_upper_limb, soft_tissue_lower_limb, lymph_node_cervical, lymph_node_supraclavicular, lymph_node_axillary, lymph_node_internal_mammary, lymph_node_mediastinal, lymph_node_hilar, lymph_node_abdominal, lymph_node_retroperitoneal, lymph_node_pelvic, lymph_node_inguinal, vessel_aorta, vessel_ivc, vessel_svc, vessel_portal_vein, vessel_hepatic_vein, vessel_hepatic_artery, vessel_sma, vessel_smv, vessel_splenic_vein, vessel_renal_vein, vessel_renal_artery, vessel_pulmonary_artery, vessel_carotid, vessel_iliac, vessel_other |
| LesionStatus | cancerous, uncertain |
| LesionNature | primary, metastasis, not_stated_unclear |
| LesionMorphology | solid, cystic, mixed_solid_cystic, ground_glass_nodule, part_solid_ground_glass, non_mass_enhancement |
| LesionInternalFeature | necrotic, calcified, haemorrhagic, mucinous, fat_containing, sclerotic, lytic |
| LesionMargin | circumscribed_welldefined, microlobulated, obscured, indistinct_illdefined, spiculated, irregular, lobulated |
| LesionShape | oval, round, irregular, lobulated |
| LesionUptake | avid, non_avid, equivocal |
| NonCancerFindingType | other, thrombus, aneurysm, dissection, stenosis_occlusion, ascites, effusion, active_haemorrhage, haematoma, abscess_pus_collection, cyst_benign_collection, pneumothorax_pneumoperitoneum, inflammation_infection, obstruction_dilatation, calculus, fibrosis_scarring, steatosis_fatty_change, atrophy, volume_loss_collapse, fracture, perforation_rupture, herniation, degenerative_change |

## License

This project uses a proprietary license issued by Guy's and St Thomas' NHS Foundation Trust, enabling free (non-commercial) use by NHS organisations. See LICENSE files for details.
