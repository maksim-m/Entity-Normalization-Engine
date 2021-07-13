from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifier, \
    SentenceTransformerAndClassifierResult
from utils import PROJECT_ROOT


def print_probs_for_classes(classification_result: np.ndarray):
    print("Address      : ", "{0:0.2f}".format(classification_result[0]))
    print("Company Name : ", "{0:0.2f}".format(classification_result[1]))
    print("Location     : ", "{0:0.2f}".format(classification_result[2]))
    print("Physical good: ", "{0:0.2f}".format(classification_result[3]))
    print("Serial number: ", "{0:0.2f}".format(classification_result[4]))


if __name__ == "__main__":

    base_model = "sentence-transformers/paraphrase-mpnet-base-v2"
    model = SentenceTransformerAndClassifier(base_model, n_classes=5)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.load_state_dict(torch.load(Path.joinpath(PROJECT_ROOT, "classification_model.pt")))
    while True:
        user_input = input("Enter next entity: ")
        encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
        model_output: SentenceTransformerAndClassifierResult = model.encode_and_classify(**encoded_input)
        classification_result = model_output.classification_result.cpu().detach().numpy()
        print_probs_for_classes(classification_result[0])
