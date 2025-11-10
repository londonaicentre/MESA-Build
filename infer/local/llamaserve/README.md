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

- (CLI) Start the server as follows: `llamaserve`.

## License

This project uses the CC BY-NC-ND 4.0 license (see [LICENSE](LICENSE)).

The contents of this repository are designed for NHS organisations to use on private data.