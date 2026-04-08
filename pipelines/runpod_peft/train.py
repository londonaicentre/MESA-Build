"""
train.py

HuggingFace PEFT LoRA fine-tuning script for RunPod
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from utils.config import load_config


def main():
    print("Loading configuration from config.yaml...")
    config = load_config()

    model_name = config["model"]
    data_dir = config["data_dir"]
    train_file = config["train_file"]
    valid_file = config.get("valid_file")

    model_cfg = config.get("model_config", {})
    lora_cfg = config["lora_config"]
    training_cfg = config["training_args"]

    print(f"Model: {model_name}")
    print(f"LoRA rank: {lora_cfg['r']}, alpha: {lora_cfg['lora_alpha']}")
    print(f"Training batch size: {training_cfg['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {training_cfg['gradient_accumulation_steps']}")
    print(f"Max sequence length: {training_cfg.get('max_length', 2048)}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print("Loading base model...")
    torch_dtype = torch.bfloat16 if training_cfg.get("bf16", False) else torch.float16
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
    }
    if "attn_implementation" in model_cfg:
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
        print(f"Using attention: {model_cfg['attn_implementation']}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if training_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    # Apply LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading training data...")
    train_path = os.path.join(data_dir, train_file)
    data_files = {"train": train_path}

    if valid_file:
        valid_path = os.path.join(data_dir, valid_file)
        data_files["validation"] = valid_path

    dataset = load_dataset("json", data_files=data_files)
    print(f"Train examples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"Validation examples: {len(dataset['validation'])}")

    training_args = SFTConfig(**training_cfg)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])

    print(f"Adapters saved to: {training_cfg['output_dir']}")

if __name__ == "__main__":
    main()
