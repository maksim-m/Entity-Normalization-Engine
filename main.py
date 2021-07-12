from pathlib import Path
from typing import Dict

import torch
from envparse import env

from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifierResult
from clustering.EntityProcessor import EntityProcessorType
from clustering.SentenceEmbeddingEntityProcessor import SentenceEmbeddingEntityProcessor
from clustering.StringEntityProcessor import StringEntityProcessor
from utils import load_model, load_class2label, inverse_dict

MODEL_PATH_STR = env.str("MODEL_PATH", default="classification_model.pt")
MODEL_PATH = Path(MODEL_PATH_STR)

CLASS2LABEL_PATH_STR = env.str("CLASS2LABEL_PATH", default="index2label.json")
CLASS2LABEL_PATH = Path(CLASS2LABEL_PATH_STR)

if __name__ == "__main__":
    print("Entity Normalization Engine")
    print("-" * 27)
    print("Usage: Type your entity and press Enter. Repeat until all entities are processed.")
    print("Type \"stop\" to stop program execution.\n")

    print("Loading model...")
    class2label = load_class2label(CLASS2LABEL_PATH)
    label2class = inverse_dict(class2label)
    model, tokenizer = load_model(MODEL_PATH)
    print("Loading model... Done")

    processors: Dict[int, EntityProcessorType] = dict()
    for class_id in class2label.keys():
        label = class2label[class_id]
        processor = StringEntityProcessor(label) if label == "serial_number" else SentenceEmbeddingEntityProcessor(label,
                                                                                                                   distance=1)
        processors[class_id] = processor

    while True:
        user_input = input("Enter next entity: ")
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
