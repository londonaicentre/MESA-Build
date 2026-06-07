"""
post_process.py

Reference post-processing of a completed SageMaker job using HuggingFaceLoRATrainer.

This script is written as a tutorial for the SECOND half of the AWS flow, turning
the raw trained adapter sitting in S3 into a versioned, merged model uploaded to S3.

Unlike the MLX flow, AWS splits training and post-processing into two scripts.
run_example.py launches the (asynchronous) SageMaker job and returns.
This script merges and uploads, once that job has completed.

Run this after run_example.py has launched a job and that job has reached status
Completed in the AWS SageMaker console. Pass the --s3-output-path and --job-name
printed by run_example.py:

    uv run python post_process.py \\
        --s3-output-path jobs/train/<job_id>/output \\
        --job-name mesa-<job_id>-...
"""

import argparse

from finetune import HuggingFaceLoRATrainer
from finetune._common_utils import build_model_card
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

# The training batch the job was fine-tuned on, found in S3:
# s3://aicentre-nlpteam-mesa-build/trainingdata/<batch_name>/
# Must match the batch(es) used in run_example.py so the reconstructed trainer and the
# model card's training_data reference the same source data.
# Multiple training batches can be specified as a list[str]

TRAINING_BATCH = "20260123-094248_test-batch"


def main(s3_output_path: str, job_name: str):
    ############################################################################################
    # 1. Reconstruct the trainer.
    #
    # There is no shared state between run_example.py and this script (they are separate
    # processes, and could be run on different machines). So we rebuild the trainer with the same
    # static args used to launch the job, keeping these identical to run_example.py.
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
        description="tutorial model using oncoschema and qwen model",
        instance_type="ml.g5.xlarge",
    )

    ############################################################################################
    # 2. Build the model card (and choose the version).
    #
    # The model card is the metadata record that travels with the model. Crucially, the
    # major/minor/patch numbers you set here are entered MANUALLY, there is no auto-increment
    # and no lookup of existing versions in S3. These three numbers are carried into the model
    # folder naming at upload time: the model is published under
    #
    #     models/<model_name>/<model_name>_<major>_<minor>_<patch>/
    #
    # so with the values below that is:
    #
    #     models/qwen-onco-example/qwen-onco-example_1_0_0/
    #
    # Re-running with the same model_name and same version numbers OVERWRITES the artifacts at
    # that prefix, so bump the version yourself when you want to keep a previous build.
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

    ############################################################################################
    # 3. Merge and publish.
    #
    # `post_process` downloads the trained LoRA adapter from `s3_output_path` (the SageMaker
    # output location), merges it into the base model, then uploads the unpacked merged model
    # plus model_card.yaml to the build bucket.
    #
    # It is possible to publish models to the MESA public bucket where it is available for deployment
    # Set push_public=True to upload to public distribution bucket as a tarball, found at:
    # s3://aicentre-nlpteam-mesa-public/.
    #
    # This is where the specific Sagemaker job args are used:
    # --s3_output_path is where SageMaker wrote its output in S3
    # --job_name is the SageMaker name for the job
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
