from dataclasses import dataclass, field
from typing import List, Optional

from torch import Tensor


@dataclass
class Entity(object):
    representative: Optional[Tensor] = None
    synonyms: List[str] = field(default_factory=list)

    def add_synonym(self, synonym: str):
        if synonym.strip() not in self.synonyms:
            self.synonyms.append(synonym.strip())
