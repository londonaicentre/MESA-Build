"""
run_example.py

Reference fine-tuning launch on AWS/SageMaker using HuggingFaceLoRATrainer.

This script is written as a tutorial for launching a SageMaker fine-tuning job
from raw training data, ready to be turned into a versioned, merged model.

Unlike the MLX flow, AWS training runs *remotely* and *asynchronously*.
This script only stages the data and launches the job, then returns.
Once the job completes, run post_process.py to merge and upload the model.
"""

from finetune import HuggingFaceLoRATrainer
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
    # This wires together four parameters needed for an AWS fine-tuning run:
    #   - `schema` + `prompt_builder`: schema gives target structure and prompt template
    #     used to turn each document into a (prompt, completion) training pair.
    #   - `training_batch_names`: which registered batch(es) to pull from the build bucket.
    #     as specified above in TRAINING_BATCH
    #   - `config_path`: training hyperparameters
    #     generally keep in config.yaml in same folder as fine-tune script
    #   - `aws_config`: the build bucket + region used for reading data and writing the model.
    #     for AWS, also needs `role`, the SageMaker execution role the job runs under.
    #     generally, keep to these defaults
    #   - `instance_type`: the EC2 instance SageMaker provisions for the job,
    #     billable for as long as the job runs.
    #
    # `model_name` is the key identifier for the model family, that also becomes the top-level
    # folder in S3 (models/<model_name>/...). It is carried through to post_process.py.
    #
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
    # 2. Prepare and stage the training data.
    #
    # `prepare_data()` downloads + validates the batch(es) and writes train.jsonl, then stages
    # it to S3 so SageMaker can read it. It returns the S3 input path the job will be pointed at.
    # To peek behind the hood, HuggingFaceLoRATrainer is defined in `finetune/hf_estimator.py`
    s3_path = trainer.prepare_data()
    print(f"Prepared + uploaded training data to: {s3_path}")

    ############################################################################################
    # 3. Launch the (billable) SageMaker job.
    #
    # This kicks off training in the cloud and returns immediately — the job runs
    # asynchronously, so this script does not wait for it to finish. `job_name` identifies the
    # job in the SageMaker console and is needed by post_process.py later.
    job_name = trainer.launch_job(s3_path)
    print(f"Launched SageMaker job: {job_name}")

    ############################################################################################
    # 4. Hand off to post-processing.
    #
    # Because the job runs asynchronously, the merge/version/upload step is a separate run.
    # Once the job shows 'Completed' in the console, invoke post_process.py with the two values
    # printed below: the S3 output path (where SageMaker writes the trained adapter) and the
    # job name. post_process.py reconstructs the trainer with the same args and finishes the flow.
    print(
        "\nMonitor the job in the AWS SageMaker console. Once its status is 'Completed', run:\n"
        f"\n  uv run python post_process.py \\\n"
        f"    --s3-output-path {trainer.s3_output_path} \\\n"
        f"    --job-name {job_name}\n"
    )


if __name__ == "__main__":
    main()
