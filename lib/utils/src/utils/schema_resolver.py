import logging
import subprocess
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from re import match
from shutil import which
from tempfile import TemporaryDirectory
from types import ModuleType

import httpx

logger = logging.getLogger(__name__)


class SchemaResolver:
    """Resolves, installs and imports the schema package serving a model."""

    __PACKAGE_PATTERN: str = r"londonaicentre-[a-z]+schema\Z"
    __ORGANISATION: str = "londonaicentre"

    @staticmethod
    def resolve_remote_schema_info(
        openai_endpoint: str, model_name: str
    ) -> tuple[str, str]:
        """Query the remote vLLM backend for the schema serving the given model.

        Args:
            openai_endpoint(str): Base URL of the OpenAI-compatible vLLM endpoint.
            model_name(str): Name of the served model to look up.

        Returns:
            tuple[str, str]: The schema name and schema version reported by the backend.
        """
        response = httpx.get(f"{openai_endpoint}/models", timeout=30.0)
        response.raise_for_status()
        model = next(
            entry for entry in response.json()["data"] if entry["id"] == model_name
        )
        return model["schema_name"], model["schema_version"]

    @staticmethod
    def install_schema_package(
        schema_name: str, schema_version: str, use_latest: bool = False
    ) -> str:
        """Install the schema package, either pinned or at the latest published version.

        Args:
            schema_name(str): Distribution package name of the schema, e.g.
                'londonaicentre-oncoschema'.
            schema_version(str): Exact version of the schema package to install. Ignored
                if use_latest is True.
            use_latest(bool): Install the latest published version instead of
                schema_version, overriding the version pinned by the model/backend.
                Defaults to False.

        Returns:
            str: The distribution package name of the installed schema.

        Raises:
            ValueError: If the package is not a londonaicentre schema package owned by
                the londonaicentre organisation on PyPI.
        """
        if use_latest:
            log_message = f"Installing latest schema package '{schema_name}'"
            pip_args = ["--upgrade-package", schema_name, schema_name]
        else:
            try:
                installed_version = version(schema_name)
            except PackageNotFoundError:
                installed_version = None
            if installed_version == schema_version:
                return schema_name
            log_message = f"Installing schema package '{schema_name}=={schema_version}'"
            pip_args = [f"{schema_name}=={schema_version}"]

        SchemaResolver.__check_published_by_organisation(schema_name)
        logger.info(log_message)
        SchemaResolver.__scan(pip_args[-1])
        subprocess.run(["uv", "pip", "install", *pip_args], check=True)
        return schema_name

    @staticmethod
    def __scan(requirement: str) -> None:
        if which("trivy") is None:
            logger.warning(f"Trivy not found, so '{requirement}' is not scanned")
            return
        with TemporaryDirectory() as directory:
            requirements: str = str(Path(directory) / "requirements.txt")
            subprocess.run(
                ["uv", "pip", "compile", "-", "-o", requirements],
                input=requirement,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    "trivy",
                    "fs",
                    "--scanners",
                    "vuln",
                    "--ignore-unfixed",
                    "--exit-code",
                    "1",
                    requirements,
                ],
                check=True,
            )

    @staticmethod
    def __check_published_by_organisation(schema_name: str) -> None:
        if not match(SchemaResolver.__PACKAGE_PATTERN, schema_name):
            raise ValueError(f"'{schema_name}' is not a londonaicentre schema package")
        response = httpx.get(f"https://pypi.org/pypi/{schema_name}/json", timeout=30.0)
        response.raise_for_status()
        if (response.json().get("ownership") or {}).get(
            "organization"
        ) != SchemaResolver.__ORGANISATION:
            raise ValueError(
                f"'{schema_name}' is not owned by the "
                f"{SchemaResolver.__ORGANISATION} organisation on PyPI"
            )

    @staticmethod
    def import_schema_modules(schema_name: str) -> tuple[ModuleType, ModuleType]:
        """Import the schema module and its prompt builder submodule.

        Args:
            schema_name(str): Distribution package name of the schema, e.g.
                'londonaicentre-oncoschema'.

        Returns:
            tuple[ModuleType, ModuleType]: The schema module and its prompt builder module.
        """
        module_name = schema_name.removeprefix("londonaicentre-")
        return import_module(module_name), import_module(
            f"{module_name}.prompt_builder"
        )
