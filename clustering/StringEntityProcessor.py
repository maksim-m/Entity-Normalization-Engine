from typing import Pattern, Optional

from pyxdameraulevenshtein import damerau_levenshtein_distance

from clustering.EntityProcessor import EntityProcessor
from clustering.entity import Entity


class StringEntityProcessor(EntityProcessor):
    def __init__(self, entity_type: str, ignored_characters: Optional[Pattern] = None, distance: int = 0,
                 ignore_case: bool = False):
        super().__init__(entity_type)
        self.ignored_characters = ignored_characters
        self.distance = distance
        self.ignore_case = ignore_case

    def _equals(self, reference_entity: Entity, candidate_input_string: str, **kwargs) -> bool:
        if self.ignore_case:
            candidate_input_string = candidate_input_string.lower()
        for synonym in reference_entity.synonyms:
            if self.ignore_case:
                synonym = synonym.lower()
            match_found = synonym == candidate_input_string if self.distance == 0 else damerau_levenshtein_distance(synonym,
                                                                                                                    candidate_input_string) <= self.distance
            if match_found:
                return True
        return False
