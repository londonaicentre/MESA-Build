"""
run_example.py

Reference end-to-end LoRA fine-tuning flow on local Apple Silicon using MLXLoRATrainer.

This script is written as a tutorial for a full MLX run from raw training data
to a versioned, merged model uploaded to S3.

Unlike the AWS flow, MLX trains and fuses *locally*.
Everything below happens on your machine.
A fused model is uploaded at the end.
"""

from finetune import MLXLoRATrainer
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

# The training batch to fine-tune on, found in S3:
# s3://aicentre-nlpteam-mesa-build/trainingdata/<batch_name>/
# This is a small 10-sample oncoschema batch so the example runs quickly.
# Multiple training batches can be specified as a list[str]

TRAINING_BATCH = "20260123-094248_test-batch"


def main():
    ############################################################################################
    # 1. Construct the trainer.
    #
    # This wires together four parameters needed for an MLX fine-tuning run:
    #   - `schema` + `prompt_builder`: schema gives target structure and prompt template
    #     used to turn each document into a (prompt, completion) training pair.
    #   - `training_batch_names`: which registered batch(es) to pull from the build bucket.
    #     as specified above in TRAINING_BATCH
    #   - `config_path`: training hyperparameters
    #     generally keep in config.yaml in same folder as fine-tune script
    #   - `aws_config`: the build bucket + region used for reading data and writing the model.
    #     generally, keep to these defaults
    #
    # `model_name` is the key identifier for the model family, that also becomes the top-level
    # folder in S3 (models/<model_name>/...).
    #
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
        description="tutorial model using oncoschema and qwen model",
        work_dir="data/models",
        quantize=None,  # Generally, use `quantize=None` to keep the fused model at full precision.
    )

    ############################################################################################
    # 2. Train locally.
    #
    # `run()` performs the full local pipeline: prepare_data (download + validate the batch and
    # write train.jsonl) -> _write_config -> invoke `mlx_lm.lora` to train the LoRA adapter.
    # Nothing is staged to S3 during training — MLX works entirely on local disk under work_dir.
    # To peek behind the hood, MLXLoRATrainer is defined in `finetune/mlx_trainer.py`
    trainer.run()

    ############################################################################################
    # 3. Build the model card (and choose the version).
    #
    # The model card is the metadata record that travels with the model. Crucially, the
    # major/minor/patch numbers you set here are entered MANUALLY, there is no auto-increment.
    # These three numbers are carried into the model folder naming at upload time: the model is
    # published under
    #
    #     models/<model_name>/<model_name>_<major>_<minor>_<patch>/
    #
    # so with the values below that is:
    #
    #     models/qwen-onco-mlx-example/qwen-onco-mlx-example_1_0_0/
    #
    # Re-running with the same model_name and same version numbers OVERWRITES the artifacts at
    # that prefix, so bump the version yourself when you want to keep a previous build.
    card = trainer.build_model_card(1, 0, 0)

    ############################################################################################
    # 4. Fuse and publish.
    #
    # `post_process` fuses the trained adapter into the base model on disk to produce a single
    # merged model, then uploads the unpacked merged model plus model_card.yaml to the build bucket
    #
    # It is possible to publish models to the MESA public bucket where it is available for deployment
    # Set push_public=True to upload to public distribution bucket as a tarball, found at:
    # s3://aicentre-nlpteam-mesa-public/.
    trainer.post_process(card, push_public=False)
    print("Post-processing complete — merged model uploaded to the build bucket.")


if __name__ == "__main__":
    main()
