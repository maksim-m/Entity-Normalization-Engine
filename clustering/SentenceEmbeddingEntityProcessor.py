from pyxdameraulevenshtein import damerau_levenshtein_distance
from sentence_transformers import util
from transformers import AutoTokenizer

from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifier, \
    SentenceTransformerAndClassifierResult
from clustering.EntityProcessor import EntityProcessor
from clustering.entity import Entity


class SentenceEmbeddingEntityProcessor(EntityProcessor):
    def __init__(self, entity_type: str, hard_threshold: float = 0.85, distance: int = 0,
                 min_length_for_typo_detection: int = 5):
        super().__init__(entity_type)
        self.hard_threshold = hard_threshold
        self.distance = distance
        self.min_length_for_typo_detection = min_length_for_typo_detection

    def _equals(self, reference_entity: Entity, candidate_input_string: str, **kwargs) -> bool:
        sentence_embeddings = kwargs["sentence_embeddings"]
        cosine_similarity = util.pytorch_cos_sim(sentence_embeddings, reference_entity.representative)
        cosine_similarity = cosine_similarity.cpu().detach().numpy()[0][0]
        if cosine_similarity > self.hard_threshold:
            return True

        long_enough_for_typo_detection = len(candidate_input_string) >= self.min_length_for_typo_detection and len(
            reference_entity.canonical_representation) >= self.min_length_for_typo_detection
        if self.distance > 0 and long_enough_for_typo_detection and damerau_levenshtein_distance(
                reference_entity.canonical_representation, candidate_input_string) <= self.distance:
            return True

        return False


# just for testing
if __name__ == "__main__":
    model = SentenceTransformerAndClassifier("sentence-transformers/paraphrase-mpnet-base-v2", n_classes=5)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
    entityClustering = SentenceEmbeddingEntityProcessor("test_type", distance=1)

    while True:
        user_input = input("Enter next entity: ")
        encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
        model_output: SentenceTransformerAndClassifierResult = model.encode_and_classify(**encoded_input)
        entityClustering.process(user_input, sentence_embeddings=model_output.sentence_embeddings)
