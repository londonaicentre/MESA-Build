# Llama serve

Serve llama models locally.

- ⬇️ Downloads weights from S3

- 📦 Unpacks

- 🚀 Serves via a local OpenAI-compatible server

## Prerequisites

### Software

- Python 3.12

### Hardware

- A GPU with >=24GB VRAM (tested on NVIDIA A30)

### Configuration

- Create a file called `.env` in the directory where you intend to run this package.
Populate it with the details you have been provided with in the following format:

```text
MODEL_NAME=
WEIGHTS_ID=
WEIGHTS_KEY=
```

## Installation

1. (Recommended) Create a virtual environment and activate it:

    ```python
    python -m venv .venv
    source .venv/bin/activate
    ```

2. Install this package: `pip install londonaicentre-llama-serve`.

## Usage

### CLI

1. Note command line arguments:

    | Argument  | Description  |
    |---|---|
    | -v, --verbose | Enable debug output (optional) |

2. Start the server as follows: `llamaserve [args]`.

## Clients

### OpenAI (example)

1. Interact with the server using the [OpenAI client](https://pypi.org/project/openai) in python:

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:5000/v1",
        api_key="blank" 
    )

    response = client.chat.completions.create(
        model="<MODEL_NAME>",
        messages=[
            {"role": "system", "content": "You are an LLM named gpt-4o"},
            {"role": "user", "content": "Hello"}
        ]
    )

    print(response.choices[0].message.content)
    ```

## License

This project uses a proprietary license (see [LICENSE](LICENSE.md)).
