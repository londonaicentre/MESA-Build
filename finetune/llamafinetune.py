import sys
import os
import json
from dotenv import load_dotenv
from sagemaker.jumpstart.estimator import JumpStartEstimator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from assets.prompts.prompts import generate_system_prompt
from utils.utils import load_config


def generate_template_file() -> None:
    with open("template.json", "w") as outfile:
        print(json.dumps({
            "prompt": generate_system_prompt("systemprompt_finetune.md") + "\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n", "completion": "{response}"
        }), 
        file=outfile)

def generate_train_file() -> None:
    with open("train.jsonl", "w") as outfile:
        with open("anthropic_batch_job.jsonl.out", "r") as infile:
            for line in infile:
                line = json.loads(line)
                text = line["modelOutput"]["content"][0]["text"].replace("```json", "").replace("```", "")
                try:
                    text = json.loads(text)
                    print(json.dumps({
                        "instruction": "Extract the given information into a structured schema.",
                        "context": text["content"],
                        "response": json.dumps(text["output"], ensure_ascii=True)
                    }), 
                    file=outfile)
                except Exception:
                    pass

def run_finetune_pipeline(
        model_id, 
        model_version, 
        role, 
        input_path, 
        output_path, 
        instance_type, 
        region, 
        instruction_tuned=True, 
        quantized=True
    ) -> None:
    estimator = JumpStartEstimator(
        model_id=model_id,
        model_version=model_version,
        role=role,
        disable_output_compression=True,
        output_path=output_path,
        instance_type=instance_type,
        region=region,
        environment={'accept_eula': 'true'},
    )
    estimator.set_hyperparameters(
        instruction_tuned='True' if instruction_tuned else 'False',
        chat_dataset='False' if instruction_tuned else 'True',
        int8_quantization='True' if quantized else 'False',
        enable_fsdp='False' if quantized else 'True'
    )
    estimator.fit({'training': input_path})

if __name__ == "__main__":
    generate_template_file()
    generate_train_file()

    config = load_config()
    load_dotenv()

    run_finetune_pipeline(
        config["llama"]["model"],
        config["llama"]["version"],
        os.environ["ROLE"],
        os.environ["INPUT_PATH"],
        os.environ["OUTPUT_PATH"],
        os.environ["INSTANCE_TYPE"],
        config["llama"]["region"]
    )

    