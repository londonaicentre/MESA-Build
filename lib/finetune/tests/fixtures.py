from pydantic import BaseModel


class SchemaFixture(BaseModel):
    foo: str
    bar: int
