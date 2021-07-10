import logging
from typing import List

import torch
from sentence_transformers import util
from torch import Tensor
from transformers import AutoTokenizer

from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifier, \
    SentenceTransformerAndClassifierResult
from entity import Entity

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s \n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

THRESHOLD = 0.85


class EntityClustering(object):

    def __init__(self):
        self.n_clusters = 0
        self.entities: List[Entity] = []

    def process(self, input_string: str, sentence_embeddings: Tensor):
        for entity in self.entities:
            cosine_similarity = util.pytorch_cos_sim(sentence_embeddings, entity.representative)
            cosine_similarity = cosine_similarity.cpu().detach().numpy()[0][0]

            if cosine_similarity > THRESHOLD:
                logger.debug("Similarity: " + str(cosine_similarity))
                if input_string.strip() not in entity.synonyms:
                    entity.synonyms.append(input_string.strip())
                print("This entity already exists. Known synonyms: " + str(entity.synonyms))
                return

        print("This is a new entity")
        self.n_clusters += 1

        entity = Entity()
        entity.representative = sentence_embeddings
        entity.synonyms.append(input_string)
        self.entities.append(entity)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.debug("CUDA available: " + str(torch.cuda.is_available()))

    model = SentenceTransformerAndClassifier("sentence-transformers/paraphrase-mpnet-base-v2", n_classes=5)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
    entityClustering = EntityClustering()

    while True:
        user_input = input("Enter next entity: ")
        encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
        model_output: SentenceTransformerAndClassifierResult = model.encode_and_classify(**encoded_input)
        entityClustering.process(user_input, model_output.sentence_embeddings)
