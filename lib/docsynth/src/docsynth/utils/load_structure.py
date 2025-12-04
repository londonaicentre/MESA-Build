import random

from schemallama_types.assets import SchemaLlamaAssets

"""
load_structure.py - loads in relevant structure as prompt
"""


class StructureLoader:
    def __init__(self, enabled_structures: list[str], assets: SchemaLlamaAssets):
        self.__assets: SchemaLlamaAssets = assets
        self.enabled_structures: list[str] = enabled_structures
        self.structures: dict[str, str] = {}

    def load_structures(self) -> dict[str, str]:
        self.structures = self.__assets.load_structures(self.enabled_structures)
        return self.structures

    def get_random_structure(self) -> tuple[str, str]:
        if not self.structures:
            raise ValueError("No structures loaded.")

        filename: str = random.choice(list(self.structures.keys()))
        content: str = self.structures[filename]
        return filename, content

    def format_structure_prompt(self, structure_content: str) -> str:
        lines: list[str] = ["## MIMIC THIS DOCUMENT STRUCTURE"]
        lines.append("")
        lines.append(
            "Use the following example as a close guide for the structure of the synthetic document. Mimic this example as far as possible. Closely follow how text is organised (e.g. in block text, or in subheadings and bullets, how colons are used) and the pattern of paragraphs and newlines. If the example structure is too short to capture all the content you need to generate, extend the structure in exactly the same way to make your synthetic document. The style points given above should be applied to this example structure, without materially changing it"
        )
        lines.append("")
        lines.append("```")
        lines.append(structure_content)
        lines.append("```")
        return "\n".join(lines)

    def get_structure_count(self) -> int:
        return len(self.structures)

    def get_structure_name_without_extension(self, filename: str) -> str:
        return self.__assets.get_structure_name_without_extension(filename)
