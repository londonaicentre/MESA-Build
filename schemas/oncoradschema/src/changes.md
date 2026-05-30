# oncoradschema: changes from schema_old.py to schema.py

## General

- Naming of enum values normalised to snake_case to match the house style used in `oncoschema` and `entityschema`.
- Adopted paired-field pattern `<field>` (enum) + `<field>_desc` (Optional[str], verbatim text) wherever a structured value has a corresponding free-text extract(e.g. in `oncoschema` -> `topography` + `topography_name_desc`).
- Every Field carries a `description=`. Numeric fields (`*_mm`, `volume_ml`) carry `ge=0`.
- Top-level gate booleans (`is_radiology_report`, `is_oncology_related`) to filter out-of-scope documents
- Simplified finding hierarchy to two independent siblings: `CancerLesion` and `NonCancerFinding`, each as a flat top-level list.
- Enums: some free-text enums replaced with closed enums, others consolidated to avoid ambiguity / overlap
- Types of lesions split across lesion nature and lesion anatomy
- Patient-level data lifted off per-lesion records (RECIST response, disease scores) onto a new `CancerStatus` block.
- Now is report finding centric, avoid patient-centric definition cancer diagnosis recording

## Other

### removed Laterality.MIDLINE
Risk of conflating "no laterality applies" with "the lesion straddles the midline.". Is either lateral or not.

### Topography -> AnatomicalSite
Old enum mixed primary-cancer site (patient centric, i.e. `oncoschema`) with lesion site on the scan (radiology 'coordinates'). We are mainly ineterested in latter. So `AnatomicalSite` adds specific lymph node stations, named vessels, peritoneal compartments, and bone/soft-tissue body-region subdivisions etc etc. Topography removed.

### LesionMargin into BI-RADS lexicon
Published lexicon is more defensible. Potentially overlapping terms reconciled:

    - `WELL_DEFINED` collapsed into `CIRCUMSCRIBED`
    - `ILL_DEFINED` and `INDISTINCT` collapsed into `INDISTINCT_ILLDEFINED` (synonyms).
    - `MICROLOBULATED` added (BI-RADS standard)
    - `SPICULATED` moved here from `LesionCharacter` (BI-RADS considers margin)
    - `UNKNOWN` is removed as handled by `Optional[...] = None`

### LesionCharacter replaced by LesionShape + is_infiltrative
Old enum contained different types of concepts that could overlap. Now `LesionShape` as BI-RADS contour. Infiltration has become `is_infiltrative` boolean.

### LesionMorphology updated, split LesionInternalFeature
More distinct subdivisions, categorised into two groups that are distinct dimensions

### Density enum replaced by `density: Optional[str]`
This is tricky due to multiple modalities + ways density is described. Have made decision to capture as free text for now, and consider a modality split in future that gates types of density description.

### RecommendationType removed
We'll pick up the key actions in MDM and clinical notes etc

### PrevStudy removed
Prior-imaging not consitently extractable, so we extract progression information, and rely on previous imaging reports.

### Lesion.is_node + Lesion.is_metastasis back to enums
Because of broader capture of nature of lesion (also capturing uncertainty) and anatomy, now moved back to Enums

### LesionCertainty added (CERTAIN / UNCERTAIN / UNSPECIFIED)
This captures lesions which may or may not be cancerous

### is_recist_target, is_miliary, is_infiltrative, is_vascular_invasion booleans added on CancerLesion
These are specific radiological assertions that don't fit any enum that serves broader purpose. `is_vascular_invasion` prev. had no first-class home.

### LesionSize axes renamed
`perpendicular_diameter_mm` / `third_dimension_mm` -> `x_mm` / `y_mm` / `z_mm`. `longest_diameter_mm` kept as a separate RECIST-relevant axis. `ge=0` constraint added to all numeric fields.

### DiseaseSpecificScore overhauled
Added `scoring_system: ScoringSystem` (enum) + `scoring_system_desc: Optional[str]` for verbatim

### CancerStatus block added (patient report-level summary)
RECIST response and disease scores moved here from per-`TumourFinding`.
These are patient whole report-level facts, not per-lesion

### NonTumourFinding -> NonCancerFinding with NonCancerFindingType
Have added enum for non cancer findings

### is_cancer_lesion_related boolean added on NonCancerFinding
Distinguishes incidental non-cancer findings from those caused by a cancer lesion
We're stopping short atm of needing to index every finding and refer to a lesion id. Maybe for next version.

### BaseFinding fields removed or relocated
- `number_of_lesions`, `lesion_distribution` removed -> too vague, what should the LLM be counting? Can calculate this afterwards based on per lesion inclusion.
- `vascular_changes` split into enums and vascular invasion flag
- `laterality` moved to per-lesion / per-finding level

### NOT_DESCRIBED added to LesionMargin, LesionShape, LesionMorphology + fields made required
Previously these were `Optional[...] = None`, which conflated "report does not characterise" with "model skipped the field". Now required with explicit `NOT_DESCRIBED` so the LLM has to commit to a positive assertion.
