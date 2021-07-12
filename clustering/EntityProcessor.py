from abc import ABC, abstractmethod
from typing import List, TypeVar

from clustering.entity import Entity

EntityProcessorType = TypeVar('EntityProcessorType', bound='EntityProcessor')


class EntityProcessor(ABC):
    def __init__(self, entity_type: str):
        self.entity_type = entity_type
        self.entities: List[Entity] = []

    def process(self, input_string: str, **kwargs):
        sentence_embeddings = kwargs["sentence_embeddings"]
        for entity in self.entities:
            if self._equals(entity, input_string, **kwargs):
                entity.add_synonym(input_string)
                print("This entity already exists. Known synonyms: " + str(entity.synonyms))
                return

        print("This is a new entity")
        entity = Entity(input_string)
        entity.representative = sentence_embeddings
        entity.synonyms.append(input_string)
        self.entities.append(entity)

    @abstractmethod
    def _equals(self, reference_entity: Entity, candidate_input_string: str, **kwargs) -> bool:
        raise NotImplementedError()

    def describe_entities(self):
        n_entities = len(self.entities)
        if n_entities == 0:
            print(f"No entities for type \"{self.entity_type}\"")
            return

        print(f"{n_entities} entities for type \"{self.entity_type}\":")
        for index, entity in enumerate(self.entities, 1):
            print(f"\tEntity {index}. Known synonyms: " + str(entity.synonyms))
