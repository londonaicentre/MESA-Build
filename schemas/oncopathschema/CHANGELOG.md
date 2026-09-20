# Changelog

## [0.5.0] - 2026-09-14 [EJ/JZ]

Arising from the PharosAI pathology schema review (JB, 2026-09-02).

### General structure changes

- **`CancerSpecimenFinding` -> `SpecimenFinding`.** No longer restricted to confirmed/suspected cancer to avoid excluding borderline and explicitly negative cases. `Specimen.findings` now holds every noteworthy observation.
- **`FindingStatus` gains `NOT_CANCEROUS`** (was `CANCEROUS` / `UNCERTAIN` only). Covers named benign/reactive findings, explicitly normal results, and confirmed-negative results (negative sentinel node, complete pathological response) which were previously unrecordable.
- **`GeneralSpecimenFeature` merged into `Feature`.** One feature vocabulary regardless of status. `Specimen.general_features` / `general_features_summary` removed.
- **`features` becomes `list[FeatureFinding]`** (`feature` + `FeatureStatus`: PRESENT / ABSENT / POSSIBLE / NOT_ASSESSABLE). A feature is recorded only if the report explicitly addresses it, which allows capture of explicit absence; `NOT_ASSESSABLE` covers one the report addresses but could not judge.
- **`Feature` gains** `COMEDO_NECROSIS`, `CRIBRIFORM_ARCHITECTURE`, `SOLID_ARCHITECTURE`, `MICROPAPILLARY_ARCHITECTURE`, `PAPILLARY_ARCHITECTURE`.
- **`NOT_STATED` removed where the field is optional** (`TreatmentResponseStatus`, `Differentiation`, `BiomarkerMethod`) - `None` already says it. Kept on `InvasionStatus` / `TumourNature`, where `None` means "not a tumour", and on `MarginStatus`, whose field is required.
- An invasive + in-situ report can be captured in two separate `SpecimenFinding`, each independently graded/sized. Where the report itemises multiple foci individually (own size/site/grade), each gets its own `SpecimenFinding`; foci reported only in aggregate stay one finding.

### Fields added

- **`Specimen.is_sentinel: bool`**: true if any node in the specimen was taken as a sentinel (e.g. "sentinel node x3").
- **`Specimen.is_multifocal` / `is_multicentric: bool`**: two or more discrete tumour foci in the same, or different, quadrant/region of the specimen - a specimen-wide signal recorded alongside per-focus findings, not instead of them.
- **`Specimen.treatment_response: TreatmentResponseStatus | None`**: response to prior therapy (`COMPLETE` / `PARTIAL` / `STABLE` / `PROGRESSION`), at specimen level.
- **`FeatureFinding.feature_desc: str | None`**: name of the feature as described in the report, for use when `feature = OTHER`. Brings `FeatureFinding` in line with `Biomarker`/`PathologyScore`/`AnatomicalSite`, which already carry a `*_desc` free-text companion
- **`SpecimenFinding.tumour_source: AnatomicalSite | None`**: the primary site the report states or concludes it originates from, useful for metastases

### Domain rules / prompt changes

- A finding is recorded for any noteworthy observation - cancer, benign, or a confirmed-negative result - but an entirely unremarkable specimen can remain null.
- Biomarkers are extracted only once confirmed (pending/awaited test no longer recorded).
- Biomarker status is pinned to this finding's own cell population, to avoid false positives from background populations. 
- Margins: state margin identity and distance_mm only where explicitly stated, do not infer.

### Pharos feedback deferred or rejected

- Primary constraint is schema being at limit of complexity.
- Margin CLOSE provided as option. Model must not have option to hallucinate or infer a status.
- 'One finding per tumour focus' adopted only where the report itemises foci individually; foci reported in aggregate (e.g. "multifocal, largest 22mm") still collapse to one finding plus the `is_multifocal`/`is_multicentric` flag, to reduce output size and the risk of hallucinated information assignation to a focus the report didn't separately describe.
- Rejected a dedicated `BORDERLINE_MALIGNANT_POTENTIAL` status and an IHC/ISH pending-result split: both asked the model to infer, or draw a distinction in an edge case. Folded into `UNCERTAIN` / `NOT_CANCEROUS`, and into "only extract confirmed results", respectively.
- Full neoadjuvant block not added, `ScoreName` carries `RESIDUAL_CANCER_BURDEN` / `TUMOUR_REGRESSION_GRADE`.
- Numeric size fields, structured TNM, structured margin identity (a `margin_name` enum) - kept as free text per MESA conventions for fields that require disambiguation, but subsequently easy to parse (precision / recall trade off); resolved instead by a prompt rule against inferring status/distance not in the text.
- Biomarker germline/somatic origin not added as considered to be an edge case, not worth additional schema surface.
