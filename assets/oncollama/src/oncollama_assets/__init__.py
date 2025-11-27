from importlib.resources import files

import yaml


class OncoLlamaAssets:
    def __init__(self):
        self.base_dir = files("oncollama_assets")

    def load_prompt_template(self, template_name="default"):
        return self.base_dir.joinpath("prompts/" + f"{template_name}.md").read_text()

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

    def load_all_profiles(self):
        self.all_profiles = []
        for item in sorted(self.base_dir.joinpath("profiles").iterdir()):
            if item.is_file() and item.name.endswith(".yml"):
                profiles = self._load_profiles_from_file(item)
                self.all_profiles.extend(profiles)
        return self.all_profiles

    def load_profiles_from_files(self, filenames):
        self.all_profiles = []
        for filename in filenames:
            file_path = "profiles/" + filename
            try:
                profiles = self._load_profiles_from_file(
                    self.base_dir.joinpath(file_path)
                )
                self.all_profiles.extend(profiles)
            except FileNotFoundError:
                raise FileNotFoundError(f"Profile file not found: {file_path}")
        return self.all_profiles

    def get_profile_count(self):
        return len(self.all_profiles)

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

    def load_style_data(self):
        return yaml.safe_load(self.base_dir.joinpath("style.yml").read_text())

    def load_content_data(self):
        return yaml.safe_load(self.base_dir.joinpath("content.yml").read_text())

    def load_structures(self, enabled_structures):
        self.structures = {}
        for filename in enabled_structures:
            file_path = "structure/" + filename
            try:
                self.structures[filename] = self.base_dir.joinpath(
                    file_path
                ).read_text()
            except FileNotFoundError:
                raise FileNotFoundError(f"Structure file not found: {file_path}")
        return self.structures

    def get_structure_name_without_extension(self, filename):
        return self.base_dir.joinpath("structure/" + filename).name
