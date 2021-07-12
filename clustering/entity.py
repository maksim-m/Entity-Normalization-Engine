from dataclasses import dataclass, field
from typing import List, Optional

from torch import Tensor


@dataclass
class Entity(object):
    representative: Optional[Tensor] = None
    synonyms: List[str] = field(default_factory=list)
