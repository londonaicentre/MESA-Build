"""
orchestrate.py

Entry point to launch SageMaker LoRA fine-tuning job
"""

from finetune import HuggingFaceLoRATrainer
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

# 1. Configure
trainer = HuggingFaceLoRATrainer(
    schema=OncologyModel,
    prompt_builder=PromptBuilder(),
    training_batch_names=["20260123-094248_test-batch"],
    hyperparameters={
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",
        "per_device_train_batch_size": 4,
        "max_seq_length": 2048,
    },
    aws_config={
        "bucket": "aicentre-nlpteam-mesa-build",
        "region": "eu-west-2",
        "role": "SagemakerExecutionRole",
    },
    description="qwen-basic-test",
    instance_type="ml.g5.xlarge",
    # ml.g5.48xlarge = 8xA10G w/ 192GB VRAM (needs >0 instances)
    # ml.g5.12xlarge = 4xA10G w/ 96GB VRAM (works w/ 0 instances)
    # A100 instances unavailable in EU without request?
)

# 2. Launch
job_name = trainer.run()
