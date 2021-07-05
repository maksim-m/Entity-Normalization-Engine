import logging
from typing import List, Dict

import torch
from sentence_transformers import util, SentenceTransformer
from torch import Tensor

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s \n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

THRESHOLD = 0.85


class EntityClustering(object):

    def __init__(self, entity_encoder):
        self.entity_encoder: SentenceTransformer = entity_encoder
        self.n_clusters = 0
        self.cluster_entities: Dict[int, List[str]] = {}
        self.cluster_representative: Dict[int, Tensor] = {}

    def process(self, entity: str):
        entity_embeddings: Tensor = self.entity_encoder.encode(entity)
        for i in range(self.n_clusters):
            cosine_similarity = util.pytorch_cos_sim(entity_embeddings, self.cluster_representative[i])
            cosine_similarity = cosine_similarity.cpu().detach().numpy()[0][0]
            if cosine_similarity > THRESHOLD:
                logger.info("Similarity: " + str(cosine_similarity))
                self.cluster_entities[i].append(entity)
                print("This entity already exists. Known synonyms: " + str(self.cluster_entities[i]))
                return

        print("This is a new entity")
        new_cluster_id = self.n_clusters
        self.n_clusters += 1

        self.cluster_entities[new_cluster_id] = [entity]
        self.cluster_representative[new_cluster_id] = entity_embeddings


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.debug("CUDA avaiable: " + str(torch.cuda.is_available()))

    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    entityClustering = EntityClustering(model)

    while True:
        user_input = input("Enter next entity: ")
        entityClustering.process(user_input)
