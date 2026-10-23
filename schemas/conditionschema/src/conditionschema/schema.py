from enum import Enum

from pydantic import BaseModel, Field


class Presence(str, Enum):
    TRUE = "True"
    FALSE = "False"
    HYPOTHETICAL = "Hypothetical"


class Subject(str, Enum):
    PATIENT = "Patient"
    OTHER = "Other"


class Time(str, Enum):
    PAST = "Past"
    RECENT = "Recent"
    FUTURE = "Future"


class Mention(BaseModel):
    value: str = Field(description="The mention, copied exactly from the document")
    presence: Presence = Field(description="Is the condition asserted?")
    subject: Subject = Field(description="To whom does the mention apply?")
    time: Time = Field(description="When does the mention apply?")


class ConditionModel(BaseModel):
    mentions: list[Mention] = Field(
        default_factory=list,
        description="Every mention of the condition in document order, repeats included",
    )
