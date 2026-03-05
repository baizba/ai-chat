from dataclasses import dataclass
from typing import Mapping, TypeAlias

Metadata: TypeAlias = Mapping[str, str | int | float | bool | None]

@dataclass
class CvDataItem:
    id: str
    document: str
    metadata: Metadata

@dataclass
class RetrievalResult(CvDataItem):
    distance: float