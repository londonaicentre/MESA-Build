# Deploy LiteLLM proxy for SageMaker AI Llama models

<img src="_assets/architecture.png" alt="isolated" width="500"/>

## Prerequisites

### Dependencies

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
- [nodejs](https://nodejs.org/en/download) (optional for formatting)

### Other

1. Generate an SSH key to be used to interface with AWS (EC2):

    ```bash
    ssh-keygen -t ed25519 -f ~/.ssh/host-aws -C "YOUR_EMAIL@example.com"
    ```

2. Reference this SSH key in a 'catch all' host entry in the SSH config file: `~/.ssh/config`:

    ```
    Host *.compute.amazonaws.com *.compute-1.amazonaws.com *.compute.internal
        User ubuntu
        IdentityFile ~/.ssh/host-aws
        IdentitiesOnly yes
    ```

## Deployment

Following the [deployment of a Llama model](https://github.com/londonaicentre/GenoLlama/tree/main/infer/demo/llamadeploy#deploy-a-llama-model) on SageMaker AI:

1. Specify details of the deployment in a new `litellm/config.yml` file, guided by the template [`litellm/config.example.yml`](litellm/config.example.yml) file. 

2. `make init` to run setup tasks

3. `make up` to deploy the proxy

4. Interact with the deployed Llama model via the proxy.

    - Interact through python:

        ```
        from openai import OpenAI

        client = OpenAI(
            base_url="http://<host>:4000", 
            api_key="<master_key>"
        )

        response = client.chat.completions.create(
            model="<model>",
            messages=[
                {"role": "system", "content": "You are an LLM named gpt-4o"},
                {"role": "user", "content": "Hello"}
            ]
        )

        print(response.choices[0].message.content)
        ```

        - `base_url="http://<host>:4000"`: host specified in the output from `terraform output -raw dns`.
        - `api_key="<master_key>"`: key specified in `config.yml` > `general_settings` > `master_key`.
        - `model="<model>",`: model name specified in `config.yml` > `model_list` > `model_name`.
    
    - Interact through a UI. A sample UI is specified in [`sample-ui`](sample-ui).
    
5. `make down` to remove the proxy 

