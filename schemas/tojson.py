import json
from schemas.genomicextractmodel import GenomicTestReport

schema = GenomicTestReport.model_json_schema()

version = '03'

with open(f"schema_v{version}.json", "w") as f:
    json.dump(schema, f, indent=4)

print("Schema generated")