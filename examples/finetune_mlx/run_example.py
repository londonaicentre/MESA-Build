"""
run_example.py

Reference end-to-end LoRA fine-tuning flow on local Apple Silicon using MLXLoRATrainer
"""

from finetune import MLXLoRATrainer
from finetune._common_utils import build_model_card
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

# 10-sample oncoschema batch
TRAINING_BATCH = "20260123-094248_test-batch"


def main() -> None:
    trainer = MLXLoRATrainer(
        schema=OncologyModel,
        prompt_builder=PromptBuilder(),
        training_batch_names=[TRAINING_BATCH],
        config_path="config.yaml",
        aws_config={
            "bucket": "aicentre-nlpteam-mesa-build",
            "region": "eu-west-2",
        },
        model_name="qwen-onco-mlx-example",
        description="qwen-onco-mlx-example",
        work_dir="data/models",
        quantize=None,
    )

    # Local: prepare_data -> _write_config -> mlx_lm.lora train.
    trainer.run()

    card = build_model_card(
        base_model=trainer.base_model,
        model_name=trainer.model_name,
        major=1,
        minor=0,
        patch=0,
        model_description=trainer.description,
        training_data=list(trainer.training_batch_names),
        output_schema=OncologyModel,
    )

    # Fuse the adapter into the base model on disk, then archive + upload the merged model.
    trainer.post_process(card)
    print("Post-processing complete — merged model uploaded to S3.")


if __name__ == "__main__":
    main()
