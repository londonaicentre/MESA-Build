from pydantic import BaseModel, Field
from typing import Optional, List, Annotated
from enum import Enum

Year = Annotated[int, Field(ge=1900, le=2100)]

class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    MIDLINE = "midline"
    NOT_APPLICABLE = "not_applicable"

class Procedure(BaseModel):
    name: str = Field(..., description="Full procedure name as documented")
    laterality: Optional[Laterality] = Field(default=None, description="Laterality of the procedure")

class Implant(BaseModel):
    device_type: Optional[str] = Field(None, description="e.g. femoral stem, pacemaker lead, mesh, stent, IOL")
    name: Optional[str] = Field(None, description="Commercial name of the implant")
    manufacturer: Optional[str] = Field(None, description="Manufacturer of the implant")
    serial_number: Optional[str] = Field(None, description="Serial number of the implant")

class OperationNote(BaseModel):
    year: Year
    indication: Optional[str] = Field(None, description="Indication for the operation")
    procedures: List[Procedure] = Field(default_factory=list, description="All procedures performed, in order documented")
    findings: Optional[str] = Field(None, description="Intraoperative or procedural findings as documented")
    implants: Optional[List[Implant]] = Field(None, description="All implants implanted, in order documented")