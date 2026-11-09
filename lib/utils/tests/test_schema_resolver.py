from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock

import httpx
import pytest
from pytest_mock import MockerFixture

from utils.schema_resolver import SchemaResolver


@dataclass
class ResolveRemoteSchemaInfoMocks:
    httpx_get: MagicMock


@dataclass
class InstallSchemaPackageMocks:
    version: MagicMock
    subprocess_run: MagicMock
    httpx_get: MagicMock


@pytest.fixture
def resolve_remote_schema_info_mocks(
    mocker: MockerFixture,
) -> ResolveRemoteSchemaInfoMocks:
    return ResolveRemoteSchemaInfoMocks(
        httpx_get=mocker.patch("utils.schema_resolver.httpx.get"),
    )


@pytest.fixture
def install_schema_package_mocks(mocker: MockerFixture) -> InstallSchemaPackageMocks:
    response: MagicMock = MagicMock()
    response.json.return_value = {"ownership": {"organization": "londonaicentre"}}
    return InstallSchemaPackageMocks(
        version=mocker.patch("utils.schema_resolver.version"),
        subprocess_run=mocker.patch("utils.schema_resolver.subprocess.run"),
        httpx_get=mocker.patch(
            "utils.schema_resolver.httpx.get", return_value=response
        ),
    )


class TestResolveRemoteSchemaInfo:
    def test_resolve_remote_schema_info_matching_model_returns_schema_info(
        self, resolve_remote_schema_info_mocks: ResolveRemoteSchemaInfoMocks
    ) -> None:
        response = MagicMock()
        response.json.return_value = {
            "data": [
                {
                    "id": "foo",
                    "schema_name": "londonaicentre-oncoschema",
                    "schema_version": "2.1.0",
                },
                {
                    "id": "bar",
                    "schema_name": "londonaicentre-barschema",
                    "schema_version": "1.0.0",
                },
            ]
        }
        resolve_remote_schema_info_mocks.httpx_get.return_value = response

        schema_name, schema_version = SchemaResolver.resolve_remote_schema_info(
            "http://foo/v1", "foo"
        )

        assert schema_name == "londonaicentre-oncoschema"
        assert schema_version == "2.1.0"
        resolve_remote_schema_info_mocks.httpx_get.assert_called_once_with(
            "http://foo/v1/models", timeout=30.0
        )
        response.raise_for_status.assert_called_once()

    def test_resolve_remote_schema_info_no_matching_model_raises_stop_iteration(
        self, resolve_remote_schema_info_mocks: ResolveRemoteSchemaInfoMocks
    ) -> None:
        response = MagicMock()
        response.json.return_value = {
            "data": [
                {
                    "id": "bar",
                    "schema_name": "londonaicentre-barschema",
                    "schema_version": "1.0.0",
                }
            ]
        }
        resolve_remote_schema_info_mocks.httpx_get.return_value = response

        with pytest.raises(StopIteration):
            SchemaResolver.resolve_remote_schema_info("http://foo/v1", "foo")

    def test_resolve_remote_schema_info_http_error_propagates(
        self, resolve_remote_schema_info_mocks: ResolveRemoteSchemaInfoMocks
    ) -> None:
        response = MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=MagicMock()
        )
        resolve_remote_schema_info_mocks.httpx_get.return_value = response

        with pytest.raises(httpx.HTTPStatusError):
            SchemaResolver.resolve_remote_schema_info("http://foo/v1", "foo")


class TestInstallSchemaPackage:
    def test_install_schema_package_not_installed_installs_pinned_version(
        self, install_schema_package_mocks: InstallSchemaPackageMocks
    ) -> None:
        install_schema_package_mocks.version.side_effect = PackageNotFoundError

        result = SchemaResolver.install_schema_package(
            "londonaicentre-oncoschema", "2.1.0"
        )

        assert result == "londonaicentre-oncoschema"
        install_schema_package_mocks.subprocess_run.assert_called_once_with(
            ["uv", "pip", "install", "londonaicentre-oncoschema==2.1.0"], check=True
        )

    def test_install_schema_package_stale_version_installs_pinned_version(
        self, install_schema_package_mocks: InstallSchemaPackageMocks
    ) -> None:
        install_schema_package_mocks.version.return_value = "2.0.0"

        result = SchemaResolver.install_schema_package(
            "londonaicentre-oncoschema", "2.1.0"
        )

        assert result == "londonaicentre-oncoschema"
        install_schema_package_mocks.subprocess_run.assert_called_once_with(
            ["uv", "pip", "install", "londonaicentre-oncoschema==2.1.0"], check=True
        )

    def test_install_schema_package_matching_version_skips_install(
        self, install_schema_package_mocks: InstallSchemaPackageMocks
    ) -> None:
        install_schema_package_mocks.version.return_value = "2.1.0"

        result = SchemaResolver.install_schema_package(
            "londonaicentre-oncoschema", "2.1.0"
        )

        assert result == "londonaicentre-oncoschema"
        install_schema_package_mocks.subprocess_run.assert_not_called()
        install_schema_package_mocks.httpx_get.assert_not_called()

    def test_install_schema_package_use_latest_ignores_pinned_version(
        self, install_schema_package_mocks: InstallSchemaPackageMocks
    ) -> None:
        result = SchemaResolver.install_schema_package(
            "londonaicentre-oncoschema", "2.1.0", True
        )

        assert result == "londonaicentre-oncoschema"
        install_schema_package_mocks.subprocess_run.assert_called_once_with(
            [
                "uv",
                "pip",
                "install",
                "--upgrade-package",
                "londonaicentre-oncoschema",
                "londonaicentre-oncoschema",
            ],
            check=True,
        )
        install_schema_package_mocks.version.assert_not_called()

    def test_install_schema_package_org_owned_checks_pypi_project(
        self, install_schema_package_mocks: InstallSchemaPackageMocks
    ) -> None:
        install_schema_package_mocks.version.side_effect = PackageNotFoundError

        SchemaResolver.install_schema_package("londonaicentre-oncoschema", "2.1.0")

        install_schema_package_mocks.httpx_get.assert_called_once_with(
            "https://pypi.org/pypi/londonaicentre-oncoschema/json", timeout=30.0
        )

    @pytest.mark.parametrize(
        "schema_name",
        [
            "foo",
            "londonaicentre-foo",
            "foo-oncoschema",
            "londonaicentre-onco1schema",
            "londonaicentre-oncoschema-foo",
        ],
    )
    def test_install_schema_package_unexpected_name_raises_value_error(
        self, install_schema_package_mocks: InstallSchemaPackageMocks, schema_name: str
    ) -> None:
        install_schema_package_mocks.version.side_effect = PackageNotFoundError

        with pytest.raises(ValueError, match="not a londonaicentre schema package"):
            SchemaResolver.install_schema_package(schema_name, "1.0.0")

        install_schema_package_mocks.httpx_get.assert_not_called()
        install_schema_package_mocks.subprocess_run.assert_not_called()

    @pytest.mark.parametrize(
        "project",
        [{"ownership": {"organization": "foo"}}, {"ownership": None}, {}],
    )
    def test_install_schema_package_not_org_owned_raises_value_error(
        self,
        install_schema_package_mocks: InstallSchemaPackageMocks,
        project: dict[str, object],
    ) -> None:
        install_schema_package_mocks.version.side_effect = PackageNotFoundError
        install_schema_package_mocks.httpx_get.return_value.json.return_value = project

        with pytest.raises(ValueError, match="not owned by the londonaicentre"):
            SchemaResolver.install_schema_package("londonaicentre-fooschema", "1.0.0")

        install_schema_package_mocks.subprocess_run.assert_not_called()

    def test_install_schema_package_unpublished_project_raises_http_status_error(
        self, install_schema_package_mocks: InstallSchemaPackageMocks
    ) -> None:
        install_schema_package_mocks.version.side_effect = PackageNotFoundError
        install_schema_package_mocks.httpx_get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=MagicMock()
        )

        with pytest.raises(httpx.HTTPStatusError):
            SchemaResolver.install_schema_package("londonaicentre-fooschema", "1.0.0")

        install_schema_package_mocks.subprocess_run.assert_not_called()


class TestImportSchemaModules:
    def test_import_schema_modules_unknown_schema_raises_module_not_found_error(
        self,
    ) -> None:
        with pytest.raises(ModuleNotFoundError):
            SchemaResolver.import_schema_modules("londonaicentre-not_a_real_schema")
