from dataclasses import dataclass, field
from typing import List

from torch import Tensor


@dataclass
class Entity(object):
    representative: Tensor = None
    synonyms: List[str] = field(default_factory=list)
