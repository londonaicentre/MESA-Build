from importlib.resources import files

import yaml

from schemallama_types.assets import SchemaLlamaAssets


class OncoLlamaAssets(SchemaLlamaAssets):
    def __init__(self):
        self.base_dir = files("oncollama_assets")

    def _load_profiles_from_file(self, file_path):
        profiles = []
        cancer_type = file_path.name
        for profile_id, profile_data in yaml.safe_load(file_path.read_text()).items():
            profiles.append(
                {
                    "profile_id": profile_id,
                    "cancer_type": cancer_type,
                    "source_file": file_path.name,
                    "morphology": profile_data.get("morphology", "UNKNOWN"),
                    "descriptive_name": profile_data.get("descriptive_name", ""),
                    "biomarker_profile": profile_data.get("biomarker_profile", ""),
                }
            )
        return profiles

    def format_profile_prompt(self, profile):
        lines = ["## USE THIS PRIMARY CANCER PROFILE"]
        lines.append("")
        lines.append(
            f"**Primary Diagnosis that should appear verbatim in document:** {profile['descriptive_name']}"
        )
        lines.append("")
        lines.append(
            f"**Biomarker Profile - Note that these are the ONLY molecular biomarker results that should be given for this patient:** {profile['biomarker_profile']}"
        )
        lines.append("")
        return "\n".join(lines)
