import random

from oncollama_assets import OncoLlamaAssets

"""
load_sampling.py - probabilistic sampling from config files into prompt
"""


class ConfigSampler:
    def __init__(self, assets: OncoLlamaAssets):
        self.style_data = assets.load_style_data()
        self.content_data = assets.load_content_data()

    def _sample_section(self, section_data):
        mutually_exclusive = section_data.get("_mutually_exclusive", False)
        items = {k: v for k, v in section_data.items() if not k.startswith("_")}

        selected = []

        if mutually_exclusive:
            choices = list(items.keys())
            weights = [items[c]["probability"] for c in choices]
            chosen = random.choices(choices, weights=weights, k=1)[0]
            description = items[chosen]["description"]
            if description:
                selected.append({"key": chosen, "description": description})
        else:
            for key, config in items.items():
                probability = config["probability"]
                if random.random() < probability:
                    description = config["description"]
                    if description:
                        selected.append({"key": key, "description": description})

        return selected

    def sample_style_config(self):
        result = {}
        for section_name, section_data in self.style_data.items():
            result[section_name] = self._sample_section(section_data)
        return result

    def sample_content_config(self):
        result = {}
        for section_name, section_data in self.content_data.items():
            result[section_name] = self._sample_section(section_data)
        return result

    def format_style_prompt(self, sampled_style):
        lines = ["## FOLLOW THESE STYLE REQUIREMENTS"]
        lines.append("")

        for section_name, items in sampled_style.items():
            if items:
                section_title = section_name.replace("_", " ").title()
                lines.append(f"**{section_title}:**")
                for item in items:
                    lines.append(f"- {item['description']}")
                lines.append("")

        return "\n".join(lines).strip()

    def format_content_prompt(self, sampled_content):
        lines = ["## FOLLOW THESE CONTENT REQUIREMENTS"]
        lines.append("")

        for section_name, items in sampled_content.items():
            if items:
                section_title = section_name.replace("_", " ").title()
                lines.append(f"**{section_title}:**")
                for item in items:
                    lines.append(f"- {item['description']}")
                lines.append("")

        return "\n".join(lines).strip()

    def generate_prompts(self):
        style_config = self.sample_style_config()
        content_config = self.sample_content_config()
        style_prompt = self.format_style_prompt(style_config)
        content_prompt = self.format_content_prompt(content_config)
        return style_prompt, content_prompt
