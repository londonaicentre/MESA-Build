# Paedacuteschema

Schema package for extracting acute triage signs ("red flags") from paediatric triage notes.

Lightweight by design: a single flat `AcuteSignType` enum whose members are the canonical
feature tokens consumed by downstream ML pipelines. Synonym/lay-phrasing guidance for
matching lives in the prompt templates rather than the enum itself.

## Structure

```text
📁 paedacuteschema
├── examples/            # Training examples showing document input and structured output
├── schema.py            # Pydantic model for specifying expected output structure
├── prompt_builder.py    # Prompt builder for data generation and inference
├── prompt_datagen.txt   # Prompt template with example (for training data generation)
├── prompt_main.txt      # Prompt template without example (for inference/deployment)
└── py.typed             # Type checking marker
```

## Usage

```python
from paedacuteschema.prompt_builder import PromptBuilder

# Initialize builder
builder = PromptBuilder()

# Build data generation prompt (with example)
datagen_prompt = builder.build_datagen_prompt()

# Build main/inference prompt (without example)
main_prompt = builder.build_main_prompt()
```

## Schema

| Type | Values |
| ---- | ------ |
| AcuteSignType | tracheal_tug, chest_recession, nasal_flaring, grunting, head_bobbing, paradoxical_breathing, apnoea, audible_wheeze, stridor, pallor, mottling, peripheral_cyanosis, central_cyanosis, prolonged_capillary_refill, cold_peripheries, petechiae, purpura, blanching_rash, non_blanching_rash, jaundice, seizure, decorticate_posturing, decerebrate_posturing, opisthotonos, bulging_fontanelle, neck_stiffness, kernigs_sign, brudzinskis_sign, photophobia, phonophobia, drowsiness, reduced_consciousness, confusion, lethargy, floppiness, inability_to_stand_or_walk, high_pitched_cry, weak_cry, inconsolable_crying, absent_social_interaction, absent_social_smile, reduced_play, poor_feeding, unable_to_drink, reduced_urine_output, joint_swelling, joint_pain, limb_pain, refusal_to_weight_bear, parent_reports_not_themselves, clinician_documents_unwell |
| Assertion | positive, negated, hypothetical, ambiguous |

## License

This project uses a proprietary license issued by Guy's and St Thomas' NHS Foundation Trust, enabling free (non-commercial) use by NHS organisations. See LICENSE files for details.
