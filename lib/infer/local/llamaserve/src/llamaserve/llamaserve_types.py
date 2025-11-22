from pydantic import BaseModel


class LlamaServeArguments(BaseModel):
    verbose: bool
