import logging

from sentence_transformers import util
from transformers import AutoTokenizer

from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifier, \
    SentenceTransformerAndClassifierResult
from clustering.EntityProcessor import EntityProcessor
from clustering.entity import Entity

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s \n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

HARD_THRESHOLD = 0.85


class SentenceEmbeddingEntityProcessor(EntityProcessor):
    def _equals(self, reference_entity: Entity, candidate_input_string: str, **kwargs) -> bool:
        sentence_embeddings = kwargs["sentence_embeddings"]
        cosine_similarity = util.pytorch_cos_sim(sentence_embeddings, reference_entity.representative)
        cosine_similarity = cosine_similarity.cpu().detach().numpy()[0][0]
        return cosine_similarity > HARD_THRESHOLD


# just for testing
if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    model = SentenceTransformerAndClassifier("sentence-transformers/paraphrase-mpnet-base-v2", n_classes=5)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
    entityClustering = SentenceEmbeddingEntityProcessor("test_type")

    while True:
        user_input = input("Enter next entity: ")
        encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
        model_output: SentenceTransformerAndClassifierResult = model.encode_and_classify(**encoded_input)
        entityClustering.process(user_input, sentence_embeddings=model_output.sentence_embeddings)
