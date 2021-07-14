import sys
from pathlib import Path
from typing import Dict

import torch
from envparse import env

from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifierResult
from clustering.EntityProcessor import EntityProcessorType
from clustering.SentenceEmbeddingEntityProcessor import SentenceEmbeddingEntityProcessor
from clustering.StringEntityProcessor import StringEntityProcessor
from utils import load_model, load_class2label, inverse_dict, clean

MODEL_PATH_STR = env.str("MODEL_PATH", default="model.pt")
MODEL_PATH = Path(MODEL_PATH_STR)

CLASS2LABEL_PATH_STR = env.str("CLASS2LABEL_PATH", default="class2label.json")
CLASS2LABEL_PATH = Path(CLASS2LABEL_PATH_STR)

BASE_MODEL = env.str("BASE_MODEL", default="sentence-transformers/paraphrase-mpnet-base-v2")


def print_header():
    print("Entity Normalization Engine")
    print("-" * 27)
    print("Usage: Type your entity and press Enter. Repeat until all entities are processed.")
    print("Type \"stop\" to stop program execution.\n")


if __name__ == "__main__":
    print_header()

    print("Loading model... ", end="")
    sys.stdout.flush()
    class2label = load_class2label(CLASS2LABEL_PATH)
    label2class = inverse_dict(class2label)
    model, tokenizer = load_model(MODEL_PATH, BASE_MODEL)
    print("Done\n")

    processors: Dict[int, EntityProcessorType] = dict()
    for class_id in class2label.keys():
        label = class2label[class_id]
        processor = StringEntityProcessor(label) if label == "serial_number" else SentenceEmbeddingEntityProcessor(label,
                                                                                                                   distance=1)
        processors[class_id] = processor

    while True:
        user_input = input("Enter next entity: ")
        user_input = clean(user_input)
        if user_input == "stop":
            break

        encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
        model_output: SentenceTransformerAndClassifierResult = model.encode_and_classify(**encoded_input)

        predicted_class = torch.argmax(model_output.classification_result, dim=1).cpu().item()
        confidence = torch.max(model_output.classification_result, dim=1)[0].cpu().item()
        print(f"Entity class: {class2label[predicted_class]} ({int(confidence * 100)}%)")

        processors[predicted_class].process(user_input, sentence_embeddings=model_output.sentence_embeddings)

    print("\nResults:\n")
    for processor in processors.values():
        processor.describe_entities()
    print("")
