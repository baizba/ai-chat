from dataclasses import dataclass
from typing import Mapping, TypeAlias

Metadata: TypeAlias = Mapping[str, str | int | float | bool | None]

@dataclass
class VectorItem:
    id: str
    document: str
    metadata: Metadata

@dataclass
class RetrievalResult(VectorItem):
    distance: float
