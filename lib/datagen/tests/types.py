from pydantic import BaseModel


class Foobar(BaseModel):
    __test__ = False
    foo: str


class TestSchema(BaseModel):
    __test__ = False
    foo: Foobar
    baz: str
    qux: list[int]
