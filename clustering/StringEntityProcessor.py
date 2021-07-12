from typing import Pattern, Optional

from pyxdameraulevenshtein import damerau_levenshtein_distance

from clustering.EntityProcessor import EntityProcessor
from clustering.entity import Entity


class StringEntityProcessor(EntityProcessor):
    def __init__(self, entity_type: str, ignored_characters: Optional[Pattern] = None, distance: int = 0):
        super().__init__(entity_type)
        self.ignored_characters = ignored_characters
        self.distance = distance

    def _equals(self, reference_entity: Entity, candidate_input_string: str, **kwargs) -> bool:
        if self.distance == 0:
            for synonym in reference_entity.synonyms:
                if synonym == candidate_input_string:
                    return True
            return False

        for synonym in reference_entity.synonyms:
            if damerau_levenshtein_distance(synonym, candidate_input_string) >= self.distance:
                return True
        return False
