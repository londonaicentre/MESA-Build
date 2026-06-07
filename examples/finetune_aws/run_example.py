"""
run_example.py

Reference fine-tuning launch on AWS/SageMaker using HuggingFaceLoRATrainer

Note: once the job is completed, run post_process.py (passing the S3 output path and
job name printed below) to download/merge the adapter, and upload the merged model to S3.
"""

from finetune import HuggingFaceLoRATrainer
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

# 10-sample oncoschema batch
TRAINING_BATCH = "20260123-094248_test-batch"

def main() -> None:
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

    # Download + validate the batch and stage train.jsonl to S3.
    s3_path = trainer.prepare_data()
    print(f"Prepared + uploaded training data to: {s3_path}")

    # Launch the (billable) SageMaker job. Returns immediately — the job runs asynchronously.
    job_name = trainer.launch_job(s3_path)
    print(f"Launched SageMaker job: {job_name}")

    print(
        "\nMonitor the job in the AWS SageMaker console. Once its status is 'Completed', run:\n"
        f"\n  uv run python post_process.py \\\n"
        f"    --s3-output-path {trainer.s3_output_path} \\\n"
        f"    --job-name {job_name}\n"
    )


if __name__ == "__main__":
    main()
