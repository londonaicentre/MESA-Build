"""
post_process.py

Post-process a completed SageMaker training job: download the trained LoRA adapter, merge it
with the base model, build a model card, and upload the merged model to S3.

Run this after run_example.py has launched a job and that job has reached status
Completed in the AWS SageMaker console. Pass the --s3-output-path and --job-name
printed by run_example.py:

    uv run python post_process.py \\
        --s3-output-path jobs/train/<job_id>/output \\
        --job-name mesa-<job_id>-...

The trainer is reconstructed with the same static args as `run_example.py`
"""

import argparse

from finetune import HuggingFaceLoRATrainer
from finetune._common_utils import build_model_card
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

TRAINING_BATCH = "20260123-094248_test-batch"


def main(s3_output_path: str, job_name: str) -> None:
    trainer = HuggingFaceLoRATrainer(
        schema=OncologyModel,
        prompt_builder=PromptBuilder(),
        training_batch_names=[TRAINING_BATCH],
        config_path="config.yaml",
        aws_config={
            "bucket": "aicentre-nlpteam-mesa-build",
            "region": "eu-west-2",
            "role": "SagemakerExecutionRole",
        },
        model_name="qwen-onco-example",
        description="qwen-onco-example",
        instance_type="ml.g5.xlarge",
    )

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

    # Download adapter -> merge with base -> upload unpacked merged model to the build bucket.
    # push_public=False keeps it off the public bucket.
    trainer.post_process(
        card, s3_output_path=s3_output_path, job_name=job_name, push_public=False
    )
    print("Post-processing complete — merged model uploaded to the build bucket.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s3-output-path",
        required=True,
        help="S3 output path prefix printed by run_example.py (jobs/train/<job_id>/output).",
    )
    parser.add_argument(
        "--job-name",
        required=True,
        help="SageMaker job name printed by run_example.py.",
    )
    args = parser.parse_args()
    main(args.s3_output_path, args.job_name)
