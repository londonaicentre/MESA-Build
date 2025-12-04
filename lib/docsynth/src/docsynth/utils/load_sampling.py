import random
from typing import Any

from schemallama_types.assets import Content, SchemaLlamaAssets, Style

"""
load_sampling.py - probabilistic sampling from config files into prompt
"""


class ConfigSampler:
    def __init__(self, assets: SchemaLlamaAssets):
        self.style_data: Style = assets.load_style_data()
        self.content_data: Content = assets.load_content_data()

    def _sample_section(self, section_data: dict[str, Any]) -> list[dict[str, str]]:
        mutually_exclusive: bool = section_data.get("mutually_exclusive", False)
        items: dict[str, Any] = {
            k: v for k, v in section_data.items() if not k.startswith("mutually_exclusive")
        }
        selected: list[dict[str, str]] = []

        description: str = ""
        if mutually_exclusive:
            choices: list[str] = list(items.keys())
            weights: list[float] = [items[c]["probability"] for c in choices]
            chosen: str = random.choices(choices, weights=weights, k=1)[0]
            description = items[chosen]["description"]
            if description:
                selected.append({"key": chosen, "description": description})
        else:
            key: str
            config: dict[str, Any]
            for key, config in items.items():
                probability: float = config["probability"]
                if random.random() < probability:
                    description = config["description"]
                    if description:
                        selected.append({"key": key, "description": description})

        return selected

    def sample_style_config(self) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = {}
        section_name: str
        section_data: dict[str, Any]
        for section_name, section_data in self.style_data.model_dump().items():
            result[section_name] = self._sample_section(section_data)
        return result

    def sample_content_config(self) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = {}
        section_name: str
        section_data: dict[str, Any]
        for section_name, section_data in self.content_data.model_dump().items():
            result[section_name] = self._sample_section(section_data)
        return result

    def format_style_prompt(
        self, sampled_style: dict[str, list[dict[str, str]]]
    ) -> str:
        lines: list[str] = ["## FOLLOW THESE STYLE REQUIREMENTS"]
        lines.append("")

        section_name: str
        items: list[dict[str, str]]
        for section_name, items in sampled_style.items():
            if items:
                section_title = section_name.replace("_", " ").title()
                lines.append(f"**{section_title}:**")
                for item in items:
                    lines.append(f"- {item['description']}")
                lines.append("")

        return "\n".join(lines).strip()

    def format_content_prompt(
        self, sampled_content: dict[str, list[dict[str, str]]]
    ) -> str:
        lines: list[str] = ["## FOLLOW THESE CONTENT REQUIREMENTS"]
        lines.append("")

        section_name: str
        items: list[dict[str, str]]
        for section_name, items in sampled_content.items():
            if items:
                section_title = section_name.replace("_", " ").title()
                lines.append(f"**{section_title}:**")
                for item in items:
                    lines.append(f"- {item['description']}")
                lines.append("")

        return "\n".join(lines).strip()

    def generate_prompts(self) -> tuple[str, str]:
        style_config: dict[str, list[dict[str, str]]] = self.sample_style_config()
        content_config: dict[str, list[dict[str, str]]] = self.sample_content_config()
        style_prompt: str = self.format_style_prompt(style_config)
        content_prompt: str = self.format_content_prompt(content_config)
        return style_prompt, content_prompt
