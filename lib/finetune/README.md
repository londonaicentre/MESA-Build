# MESA: Fine-tune

Fine-tune a MESA model on AWS Sagemaker and locally on Apple Silicon.

See [examples](examples/).

## Quickstart (AWS Sagemaker)

1. Follow the instructions [here](https://github.com/londonaicentre/sde_aic_internal_docs/blob/main/nlp/using_amazon_sagemaker.md#sso-authentication-for-use-of-the-aws-cli) to set up SSO authentication for use of the AWS CLI.

2. Start a fine-tuning run:

```python
from finetune import HuggingFaceLoRATrainer
HuggingFaceLoRATrainer(
    schema=,
    prompt_builder=,
    training_batch_names=[],
    hyperparameters={
        "base_model":,
        "num_epochs":,
        "learning_rate":,
        "lora_r":,
        "lora_alpha":,
        "lora_target_modules":,
        "per_device_train_batch_size":,
        "max_seq_length":,
    },
    aws_config={
        "bucket":,
        "region":,
        "role":,
    },
    description=,
    instance_type=,
).run()
```
