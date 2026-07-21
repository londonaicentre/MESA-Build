from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from finetune.mlx_trainer import MLXLoRATrainer
from pydantic import ValidationError
from pytest_mock import MockerFixture

from finetune_runner.__main__ import FinetuneMLXRunner


def _runner(**overrides: Any) -> FinetuneMLXRunner:
    return FinetuneMLXRunner.model_validate(
        {
            "training_batch_name": "foo",
            "model_name": "bar",
            "version": "1.2.3",
            **overrides,
        }
    )


@pytest.fixture
def runner() -> FinetuneMLXRunner:
    return _runner()


class TestLoadSchema:
    def test_installs_latest_and_returns_schema_and_prompt_builder(
        self, mocker: MockerFixture
    ) -> None:
        schema_module: MagicMock = MagicMock()
        prompt_builder_module: MagicMock = MagicMock()
        install_schema_package: MagicMock = mocker.patch(
            "finetune_runner.__main__.SchemaResolver.install_schema_package",
            return_value="londonaicentre-genoschema",
        )
        import_schema_modules: MagicMock = mocker.patch(
            "finetune_runner.__main__.SchemaResolver.import_schema_modules",
            return_value=(schema_module, prompt_builder_module),
        )
        schema, prompt_builder = _runner(schema="genoschema")._load_schema()
        install_schema_package.assert_called_once_with(
            "londonaicentre-genoschema", "", True
        )
        import_schema_modules.assert_called_once_with("londonaicentre-genoschema")
        assert schema is schema_module.Schema
        assert prompt_builder is prompt_builder_module.PromptBuilder.return_value

    def test_install_failure_propagates(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "finetune_runner.__main__.SchemaResolver.install_schema_package",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(RuntimeError):
            _runner(schema="genoschema")._load_schema()


class TestBuildValidatedModelCard:
    def test_valid_version_returns_model_card(self, runner: FinetuneMLXRunner) -> None:
        mock_trainer: MagicMock = MagicMock()
        mock_trainer.valid_model_card_version.return_value = True
        assert (
            runner._build_validated_model_card(mock_trainer)
            is mock_trainer.build_model_card.return_value
        )

    def test_invalid_version_raises(self, runner: FinetuneMLXRunner) -> None:
        mock_trainer: MagicMock = MagicMock()
        mock_trainer.valid_model_card_version.return_value = False
        with pytest.raises(ValueError):
            runner._build_validated_model_card(mock_trainer)

    def test_builds_model_card_with_version_parts(
        self, runner: FinetuneMLXRunner
    ) -> None:
        mock_trainer: MagicMock = MagicMock()
        mock_trainer.valid_model_card_version.return_value = True
        runner._build_validated_model_card(mock_trainer)
        mock_trainer.build_model_card.assert_called_once_with(1, 2, 3)


class TestValidateStage:
    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"train": True, "post_process": True}, "at most one of --train"),
            (
                {"post_process": True, "resume": True, "spec": "{}"},
                "at most one of --post-process",
            ),
            ({"post_process": True}, "--post-process requires --spec"),
            ({"resume": True}, "--resume requires --spec"),
            ({"train": True}, "requires --spec-out"),
        ],
    )
    def test_invalid_stage_raises(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            _runner(**overrides)


class TestTrain:
    def test_validates_sets_up_writes_spec_then_trains(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        validate: MagicMock = mocker.patch.object(
            FinetuneMLXRunner, "_build_validated_model_card"
        )
        trainer: MagicMock = MagicMock()
        trainer.setup.return_value = "cfg.yaml"
        trainer.to_json.return_value = '{"foo": 1}'
        spec_file: Path = tmp_path / "spec.json"
        _runner(train=True, spec_out=str(spec_file))._train(trainer)
        validate.assert_called_once_with(trainer)
        trainer.setup.assert_called_once()
        assert spec_file.read_text() == '{"foo": 1}'  # spec persisted before train
        trainer.train.assert_called_once_with("cfg.yaml")

    def test_without_spec_out_skips_spec(
        self, runner: FinetuneMLXRunner, mocker: MockerFixture
    ) -> None:
        mocker.patch.object(FinetuneMLXRunner, "_build_validated_model_card")
        write_spec: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_write_spec")
        trainer: MagicMock = MagicMock()
        trainer.setup.return_value = "cfg.yaml"
        runner._train(trainer)
        write_spec.assert_not_called()
        trainer.train.assert_called_once_with("cfg.yaml")


class TestTrainAndPostProcess:
    def test_trains_then_post_processes(
        self, runner: FinetuneMLXRunner, mocker: MockerFixture
    ) -> None:
        train: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_train")
        post: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_post_process")
        trainer: MagicMock = MagicMock()
        runner._train_and_post_process(trainer)
        train.assert_called_once_with(trainer)
        post.assert_called_once_with(trainer)


class TestResume:
    def test_not_train_resumes_then_post_processes(self, mocker: MockerFixture) -> None:
        post: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_post_process")
        trainer: MagicMock = MagicMock()
        trainer.setup.return_value = "cfg.yaml"
        _runner(resume=True, spec='{"foo": 1}')._resume(trainer)
        trainer.resume_train.assert_called_once_with("cfg.yaml")
        post.assert_called_once_with(trainer)

    def test_train_only_writes_spec_without_post_process(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        post: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_post_process")
        trainer: MagicMock = MagicMock()
        trainer.setup.return_value = "cfg.yaml"
        trainer.to_json.return_value = '{"foo": 1}'
        spec_file: Path = tmp_path / "spec.json"
        _runner(
            resume=True, train=True, spec='{"foo": 1}', spec_out=str(spec_file)
        )._resume(trainer)
        assert spec_file.read_text() == '{"foo": 1}'
        trainer.resume_train.assert_called_once_with("cfg.yaml")
        post.assert_not_called()


class TestPostProcess:
    @pytest.mark.parametrize("push_public", [False, True])
    def test_builds_card_and_post_processes(self, push_public: bool) -> None:
        trainer: MagicMock = MagicMock()
        _runner(push_public=push_public)._post_process(trainer)
        trainer.build_model_card.assert_called_once_with(1, 2, 3)
        trainer.post_process.assert_called_once_with(
            trainer.build_model_card.return_value, push_public=push_public
        )


class TestGuarded:
    def test_terminal_success_deletes(self, runner: FinetuneMLXRunner) -> None:
        trainer: MagicMock = MagicMock()
        runner._guarded(trainer, lambda _: None, True)
        trainer.cleanup.assert_called_once()

    def test_non_terminal_success_keeps(self, runner: FinetuneMLXRunner) -> None:
        trainer: MagicMock = MagicMock()
        runner._guarded(trainer, lambda _: None, False)
        trainer.cleanup.assert_not_called()

    def test_cancel_deletes_and_reraises(self, runner: FinetuneMLXRunner) -> None:
        trainer: MagicMock = MagicMock()

        def _cancel(_: MLXLoRATrainer) -> None:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            runner._guarded(trainer, _cancel, False)
        trainer.cleanup.assert_called_once()

    def test_error_keeps_and_propagates(self, runner: FinetuneMLXRunner) -> None:
        trainer: MagicMock = MagicMock()

        def _fail(_: MLXLoRATrainer) -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            runner._guarded(trainer, _fail, True)
        trainer.cleanup.assert_not_called()


class TestCliCmd:
    def test_default_guards_train_and_post_process(
        self, runner: FinetuneMLXRunner, mocker: MockerFixture
    ) -> None:
        guarded: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_guarded")
        make: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_make_trainer")
        runner.cli_cmd()
        guarded.assert_called_once_with(
            make.return_value, runner._train_and_post_process, True
        )

    def test_train_guards_train_without_delete(self, mocker: MockerFixture) -> None:
        guarded: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_guarded")
        make: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_make_trainer")
        cli: FinetuneMLXRunner = _runner(train=True, spec_out="spec.json")
        cli.cli_cmd()
        guarded.assert_called_once_with(make.return_value, cli._train, False)

    def test_post_process_guards_from_spec_with_delete(
        self, mocker: MockerFixture
    ) -> None:
        guarded: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_guarded")
        from_json: MagicMock = mocker.patch.object(MLXLoRATrainer, "from_json")
        cli: FinetuneMLXRunner = _runner(post_process=True, spec='{"foo": 1}')
        cli.cli_cmd()
        from_json.assert_called_once_with('{"foo": 1}')
        guarded.assert_called_once_with(from_json.return_value, cli._post_process, True)

    def test_resume_guards_resume_with_delete(self, mocker: MockerFixture) -> None:
        guarded: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_guarded")
        from_json: MagicMock = mocker.patch.object(MLXLoRATrainer, "from_json")
        cli: FinetuneMLXRunner = _runner(resume=True, spec='{"foo": 1}')
        cli.cli_cmd()
        guarded.assert_called_once_with(from_json.return_value, cli._resume, True)

    def test_resume_train_only_guards_resume_without_delete(
        self, mocker: MockerFixture
    ) -> None:
        guarded: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_guarded")
        mocker.patch.object(MLXLoRATrainer, "from_json")
        cli: FinetuneMLXRunner = _runner(
            resume=True, train=True, spec='{"foo": 1}', spec_out="spec.json"
        )
        cli.cli_cmd()
        assert guarded.call_args.args[2] is False
