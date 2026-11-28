import json
import socket
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from mesa_types.base_model import BaseModelReport
from pytest_mock import MockerFixture
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from utils.base_model_checker import BaseModelChecker


class BaseModelCheckerFixture(BaseModelChecker):
    @staticmethod
    def check_formats(folder: Path) -> list[str]:
        return BaseModelChecker._check_formats(folder)

    @staticmethod
    def check_remote_code(folder: Path) -> list[str]:
        return BaseModelChecker._check_remote_code(folder)

    @staticmethod
    def jail(folder: Path) -> list[str]:
        return BaseModelChecker._jail(folder)

    @staticmethod
    def generate(folder: Path) -> str:
        return BaseModelChecker._generate(folder)


@dataclass
class CheckMocks:
    _check_formats: MagicMock
    _check_remote_code: MagicMock
    _jail: MagicMock


@dataclass
class JailMocks:
    _generate: MagicMock
    logger: MagicMock


@pytest.fixture
def check_mocks(mocker: MockerFixture) -> CheckMocks:
    return CheckMocks(
        _check_formats=mocker.patch.object(
            BaseModelChecker, "_check_formats", return_value=[]
        ),
        _check_remote_code=mocker.patch.object(
            BaseModelChecker, "_check_remote_code", return_value=[]
        ),
        _jail=mocker.patch.object(BaseModelChecker, "_jail", return_value=[]),
    )


@pytest.fixture
def jail_mocks(mocker: MockerFixture) -> JailMocks:
    return JailMocks(
        _generate=mocker.patch.object(BaseModelChecker, "_generate"),
        logger=mocker.patch("utils.base_model_checker.logger"),
    )


@pytest.fixture
def model(tmp_path: Path) -> Path:
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=2,
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
    ).save_pretrained(tmp_path)
    tokenizer: Tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "Hello": 1}, "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer,
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
    ).save_pretrained(tmp_path)
    return tmp_path


class TestCheck:
    def test_check_clean_files_returns_sample_prompt_findings(
        self, check_mocks: CheckMocks
    ) -> None:
        check_mocks._jail.return_value = ["foo"]

        assert BaseModelChecker.check(Path("/bar"), "baz/qux", "quux") == (
            BaseModelReport(repo_id="baz/qux", revision="quux", findings=["foo"])
        )
        check_mocks._jail.assert_called_once_with(Path("/bar"))

    def test_check_format_findings_skips_later_checks(
        self, check_mocks: CheckMocks
    ) -> None:
        check_mocks._check_formats.return_value = ["foo"]

        assert BaseModelChecker.check(Path("/qux"), "quux/corge", "grault") == (
            BaseModelReport(repo_id="quux/corge", revision="grault", findings=["foo"])
        )
        check_mocks._check_remote_code.assert_not_called()
        check_mocks._jail.assert_not_called()

    def test_check_remote_code_findings_skips_sample_prompt(
        self, check_mocks: CheckMocks
    ) -> None:
        check_mocks._check_remote_code.return_value = ["foo"]

        assert BaseModelChecker.check(Path("/qux"), "quux/corge", "grault") == (
            BaseModelReport(repo_id="quux/corge", revision="grault", findings=["foo"])
        )
        check_mocks._check_remote_code.assert_called_once_with(Path("/qux"))
        check_mocks._jail.assert_not_called()


@pytest.mark.integration
class TestRecord:
    def test_record_clean_report_adds_revision(self, tmp_path: Path) -> None:
        (tmp_path / "records.yaml").write_text("models:\n  foo/bar: baz\n")

        BaseModelChecker.record(
            BaseModelReport(repo_id="qux/quux", revision="corge", findings=[]),
            tmp_path / "records.yaml",
        )

        assert yaml.safe_load((tmp_path / "records.yaml").read_text()) == {
            "models": {"foo/bar": "baz", "qux/quux": "corge"}
        }

    def test_record_report_with_findings_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="qux/quux has findings"):
            BaseModelChecker.record(
                BaseModelReport(repo_id="qux/quux", revision="corge", findings=["foo"]),
                tmp_path / "records.yaml",
            )


@pytest.mark.integration
class TestCheckFormats:
    def test_check_formats_allowed_files_returns_empty(self, tmp_path: Path) -> None:
        for name in ["model.safetensors", "config.json", ".gitattributes", "LICENSE"]:
            (tmp_path / name).write_text("")

        assert BaseModelCheckerFixture.check_formats(tmp_path) == []

    def test_check_formats_disallowed_files_returns_findings(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "foo").mkdir()
        for name in ["pytorch_model.bin", "foo/bar.py", "baz"]:
            (tmp_path / name).write_text("")

        assert BaseModelCheckerFixture.check_formats(tmp_path) == [
            "baz: format not allowed",
            "foo/bar.py: format not allowed",
            "pytorch_model.bin: format not allowed",
        ]


@pytest.mark.integration
class TestCheckRemoteCode:
    def test_check_remote_code_plain_configs_returns_empty(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"foo": "bar"}))

        assert BaseModelCheckerFixture.check_remote_code(tmp_path) == []

    @pytest.mark.parametrize("name", ["config.json", "tokenizer_config.json"])
    def test_check_remote_code_auto_map_returns_finding(
        self, tmp_path: Path, name: str
    ) -> None:
        (tmp_path / name).write_text(json.dumps({"auto_map": {"foo": "bar.Baz"}}))

        assert BaseModelCheckerFixture.check_remote_code(tmp_path) == [
            f"{name}: needs remote code (auto_map)"
        ]


class TestJail:
    def test_jail_offline_prompt_returns_empty(self, jail_mocks: JailMocks) -> None:
        assert BaseModelCheckerFixture.jail(Path("/foo")) == []
        jail_mocks._generate.assert_called_once_with(Path("/foo"))

    def test_jail_connection_attempt_raises(self, jail_mocks: JailMocks) -> None:
        jail_mocks._generate.side_effect = lambda _: socket.getaddrinfo(
            "foo.invalid", 80
        )

        with pytest.raises(PermissionError, match="socket.getaddrinfo"):
            BaseModelCheckerFixture.jail(Path("/bar"))
        jail_mocks.logger.warning.assert_called_once()

    def test_jail_swallowed_attempt_returns_finding(
        self, jail_mocks: JailMocks
    ) -> None:
        def generate(_: Path) -> None:
            try:
                socket.getaddrinfo("foo.invalid", 80)
            except PermissionError:
                pass

        jail_mocks._generate.side_effect = generate

        assert [
            finding.split(" (")[0]
            for finding in BaseModelCheckerFixture.jail(Path("/bar"))
        ] == ["model jail: network access"]


@pytest.mark.integration
class TestGenerate:
    def test_generate_local_model_returns_text(self, model: Path) -> None:
        assert isinstance(BaseModelCheckerFixture.generate(model), str)
