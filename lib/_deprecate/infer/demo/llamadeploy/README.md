# SchemaLlama: Deploy

Deploy a Llama model on AWS SageMaker AI

## Getting started

### AWS

- Follow the instructions [here](https://github.com/londonaicentre/sde_aic_internal_docs/blob/main/nlp/using_amazon_sagemaker.md#sso-authentication-for-use-of-the-aws-cli) to set up SSO authentication for use of the AWS CLI.

### Add custom `inference.py` to model zip

A custom [`inference.py`](inference.py) file should be added to the zipped model weights, to customise, among other things, output decoding when Llama-structured prompts are used.
It can also be used to hard code parameters such a temperature values.

## Usage

1. Deploy a llama model:

```python
from deploy.llama import run_deploy_up
run_deploy_up(
    <Bucket>, 
    <Path>, 
    <Sagemaker execution role>, 
    <Image>, 
    <Instance flavour>
):
```

### Test deployed model

1. Install LiteLLM: `pip install litellm`

2. Run the following code:

    ```
    from litellm import completion

    response = completion(
        model="sagemaker/<models.json > llama > endpoint_name>", 
        messages=[
            {"role": "system", "content": "You are an LLM named gpt-4o"},
            {"role": "user", "content": "Hello"}
        ],
        hf_model_name="<hf_model_name>"
    )

    print(response.choices[0].message.content)
    ```

    - `model="sagemaker/<endpoint_name>",`: endpoint name specified in `models.json` > `llama` > `endpoint_name`. 
    - `hf_model_name="<hf_model_name>"`: the chat template to apply to the prompt, derived from a huggingface base model. defined in: `models.json` > `llama` > `hf_model_name`.