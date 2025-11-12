# Llama serve

Serve llama models locally.

- ⬇️ Downloads weights from S3

- 📦 Unpacks

- 🚀 Serves via a local OpenAI-compatible server

## Prerequisites

- Python 3.12

### Configuration

- Create a .env file with the details you have been provided with:

```
MODEL=
WEIGHTS_ID=
WEIGHTS_KEY=
```

## Installation

1. (Recommended) Create a virtual environment and activate it: 

```
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

### OpenAI

1. Interact with the server using the [OpenAI client](https://pypi.org/project/openai/0.26.5/) in python:

    ```
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:4000",
        api_key="blank" 
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

## License

This project uses the CC BY-NC-ND 4.0 license (see [LICENSE](LICENSE)).

The contents of this repository are designed for NHS organisations to use on private data.