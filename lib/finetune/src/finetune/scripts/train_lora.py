"""
train_lora.py

SageMaker LoRA training script
Uploaded to SageMaker GPU instances for fine-tuning
https://huggingface.co/docs/sagemaker/train
"""

import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj"
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    print(f"Starting LoRA fine-tuning: {args.base_model}")

    train_data_path = os.environ["SM_CHANNEL_TRAINING"]
    print(f"Loading data from: {train_data_path}")
    dataset = load_dataset(
        "json", data_files=f"{train_data_path}/train.jsonl", split="train"
    )
    print(f"Training samples: {len(dataset)}")

    print(f"Loading model/tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    # LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    print("Trainable parameters:")
    model.print_trainable_parameters()

    # training
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,  # keep only best checkpoint
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=args.max_seq_length,
        dataset_text_field="messages",  # openAI messages format
    )

    trainer.train()

    # save
    print("Saving model...")
    trainer.save_model("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")

    print("Training complete!")
