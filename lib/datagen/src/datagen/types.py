from collections.abc import Mapping, Sequence

Scalar = str | int | float | bool | None
type JSONValue = Scalar | Sequence[JSONValue] | Mapping[str, JSONValue]
