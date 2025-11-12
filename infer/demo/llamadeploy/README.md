# Deploy a Llama model on AWS SageMaker AI

## Arrange access to SageMaker

Follow the instructions [here](https://github.com/londonaicentre/sde_aic_internal_docs/blob/main/nlp/using_amazon_sagemaker.md#sso-authentication-for-use-of-the-aws-cli) to set up SSO authentication for use of the AWS CLI.

## Add custom `inference.py` to model zip

A custom [`inference.py`](inference.py) file should be added to the zipped model weights, to customise, among other things, output decoding when Llama-structured prompts are used.
It can also be used to hard code parameters such a temperature values.

## Deploy a Llama model

Using a venv with the required packages installed (`pip install -r requirements.txt`) and after setting the right env variables, run `llamadeploy.py`.
Some command line arguments are optional, set what you need to change:

- `-p, --path`: Path within S3 bucket to the zipped weights of the model to deploy. Required.

- `[command]`: The action to perform: `up` (deploy) or `down` (delete). Required.

## Test deployed model

1. Install LiteLLM: `pip install litellm`

2. Run the following code:

    ```
    from litellm import completion

    response = completion(
        model="sagemaker/<config.json > llama > endpoint_name>", 
        messages=[
            {"role": "system", "content": "You are an LLM named gpt-4o"},
            {"role": "user", "content": "Hello"}
        ],
        hf_model_name="<hf_model_name>"
    )

    print(response.choices[0].message.content)
    ```

    - `model="sagemaker/<endpoint_name>",`: endpoint name specified in `config.json` > `llama` > `endpoint_name`. 
    - `hf_model_name="<hf_model_name>"`: the chat template to apply to the prompt, derived from a huggingface base model. defined in: `config.json` > `llama` > `hf_model_name`.