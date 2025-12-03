import random

from schemallama_types.assets import SchemaLlamaAssets

"""
load_structure.py - loads in relevant structure as prompt
"""


class StructureLoader:
    def __init__(self, enabled_structures, assets: SchemaLlamaAssets):
        self.__assets = assets
        self.enabled_structures = enabled_structures
        self.structures = {}

    def load_structures(self):
        self.structures = self.__assets.load_structures(self.enabled_structures)
        return self.structures

    def get_random_structure(self):
        if not self.structures:
            raise ValueError("No structures loaded.")

        filename = random.choice(list(self.structures.keys()))
        content = self.structures[filename]
        return filename, content

    def format_structure_prompt(self, structure_content):
        lines = ["## MIMIC THIS DOCUMENT STRUCTURE"]
        lines.append("")
        lines.append(
            "Use the following example as a close guide for the structure of the synthetic document. Mimic this example as far as possible. Closely follow how text is organised (e.g. in block text, or in subheadings and bullets, how colons are used) and the pattern of paragraphs and newlines. If the example structure is too short to capture all the content you need to generate, extend the structure in exactly the same way to make your synthetic document. The style points given above should be applied to this example structure, without materially changing it"
        )
        lines.append("")
        lines.append("```")
        lines.append(structure_content)
        lines.append("```")
        return "\n".join(lines)

    def get_structure_count(self):
        return len(self.structures)

    def get_structure_name_without_extension(self, filename):
        return self.__assets.get_structure_name_without_extension(filename)
