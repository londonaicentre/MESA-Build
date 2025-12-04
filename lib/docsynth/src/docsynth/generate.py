import json
import logging
import re
from datetime import datetime
from pathlib import Path

from docsynth.config import LLM, PipelineConfig
from schemallama_types.assets import Profile, SchemaLlamaAssets
from docsynth.utils.build_prompt import PromptBuilder
from docsynth.utils.llm_clients import LLMClient, create_llm_client

"""
generate.py - config driven synthetic document generation
"""


class Generator:
    def __init__(self) -> None:
        # basic now for debug
        logging.basicConfig(
            filename="debug.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.__logger: logging.Logger = logging.getLogger(__name__)

    def extract_output_content(self, response_text: str) -> str:
        """
        Extract content between <OUTPUT> tags
        """
        pattern: str = r"<OUTPUT>(.*?)</OUTPUT>"
        match: re.Match[str] | None = re.search(pattern, response_text, re.DOTALL)

        if match:
            content: str = match.group(1).strip()
            self.__logger.debug(
                f"Successfully extracted content from <OUTPUT> tags (length={len(content)} chars)"
            )
            return content
        else:
            self.__logger.warning(
                "No <OUTPUT> tags found in response, using full response text"
            )
            return response_text.strip()

    def save_document(
        self, output_dir: str, doc_id: str, prompt: str, content: str | None = None
    ) -> None:
        """
        Saves output document as JSON file
        If content is None, only saves prompt (debugging prompt-only mode)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        output: dict[str, str] = {
            "doc_id": doc_id,
            "doc_name": "synth",
            "prompt": prompt,
        }

        if content is not None:
            output["content"] = content

        output_path: Path = Path(output_dir) / f"{doc_id}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        self.__logger.debug(f"Saved document to {output_path}")

    def generate_doc_id(self, structure_name: str, profile_id: str) -> str:
        """
        Generate unique document ID as {structure}_{profile}_{timestamp}
        """
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        return f"{structure_name}_{profile_id}_{timestamp}"

    def generate(self, assets: SchemaLlamaAssets) -> None:
        self.__logger.info("Starting document generation pipeline")
        print("Loading pipeline.yml...")
        pipeline_config: PipelineConfig = PipelineConfig()

        print("Building prompt...")

        enabled_structures: list[str] = (
            pipeline_config.structure_selection.enabled_structures
        )

        builder: PromptBuilder = PromptBuilder(assets, enabled_structures)

        profile_files: list[str] = pipeline_config.profile_selection.file
        builder.load_profiles(profile_files)

        if profile_files:
            print(f"Loaded profiles from: {', '.join(profile_files)}")
        else:
            print("Loaded all profiles")

        print(f"Total profiles: {builder.get_profile_count()}")

        # initialise chosen LLM client
        llm_config: LLM = pipeline_config.llm
        llm_client: LLMClient | None = None

        if llm_config.enabled:
            provider: str = llm_config.provider
            try:
                print(f"Initialising LLM client (provider: {provider})...")
                llm_client = create_llm_client(llm_config)
                if llm_client:
                    print("LLM client initialised")
                    self.__logger.info(f"LLM client initialised: {provider}")
                else:
                    print("LLM generation disabled (provider set to 'none')")
            except Exception as e:
                print(f"Error initialising LLM client: {e}")
                self.__logger.error(f"Failed to initialize LLM client: {e}")
                return
        else:
            print("LLM generation disabled (saving prompts only)")
            self.__logger.info("LLM generation disabled")

        output_dir: str = "output/" + pipeline_config.output.subdirectory
        print(f"Output directory: {output_dir}")

        mode: str = str(pipeline_config.profile_selection.mode)
        count: int = pipeline_config.profile_selection.count
        include_style: bool = pipeline_config.prompt_config.include_style
        include_content: bool = pipeline_config.prompt_config.include_content

        total_docs: int = builder.get_profile_count() if count == -1 else count

        action: str = "documents" if llm_client else "prompts"
        print(f"Generating {total_docs} {action} in '{mode}' mode...")
        print("#" * 60)

        # todo: can refactor this as sequential and random share identical code
        i: int
        profile: Profile
        prompt: str
        structure_name: str
        profile_id: str
        content: str | None = None
        response: str | None
        if mode == "sequential":
            for i, profile in enumerate(builder.get_sequential_profiles(), 1):
                if i > total_docs:
                    break

                prompt, structure_name, profile_id = builder.build_prompt(
                    profile, include_style, include_content
                )
                doc_id: str = self.generate_doc_id(structure_name, profile_id)

                if llm_client:
                    try:
                        self.__logger.info(f"Generating content for {doc_id}")
                        response = llm_client.generate(prompt)
                        content = self.extract_output_content(str(response))
                        self.__logger.info(
                            f"Successfully generated content for {doc_id} (length={len(content)} chars)"
                        )
                    except Exception as e:
                        self.__logger.error(
                            f"Error generating content for {doc_id}: {e}"
                        )
                        print(f"[{i}/{total_docs}] error: {doc_id} - {e}")
                        continue

                print(f"[{i}/{total_docs}] Generated: {doc_id}")
                self.save_document(output_dir, doc_id, prompt, content)

        elif mode == "random":
            for i in range(1, total_docs + 1):
                profile = builder.get_random_profile()
                prompt, structure_name, profile_id = builder.build_prompt(
                    profile, include_style, include_content
                )
                doc_id = self.generate_doc_id(structure_name, profile_id)

                content = None
                if llm_client:
                    try:
                        self.__logger.info(f"Generating content for {doc_id}")
                        response = llm_client.generate(prompt)
                        content = self.extract_output_content(str(response))
                        self.__logger.info(
                            f"Successfully generated content for {doc_id} (length={len(content)} chars)"
                        )
                    except Exception as e:
                        self.__logger.error(
                            f"Error generating content for {doc_id}: {e}"
                        )
                        print(f"[{i}/{total_docs}] error: {doc_id} - {e}")
                        continue

                print(f"[{i}/{total_docs}] Generated: {doc_id}")
                self.save_document(output_dir, doc_id, prompt, content)

        print("#" * 60)
        print(f"Generated {total_docs} {action}")
        print(f"Saved to: {output_dir}")
        self.__logger.info(
            f"Pipeline completed successfully. Generated {total_docs} {action}"
        )
