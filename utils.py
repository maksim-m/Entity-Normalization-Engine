import io
import json
import re
from pathlib import Path
from typing import Tuple, Dict, Callable, TypeVar

import torch
from transformers import AutoTokenizer

from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifier

PROJECT_ROOT = Path(__file__).parent


def load_model(filename: Path, base_model: str) -> Tuple[SentenceTransformerAndClassifier, Callable]:
    model = SentenceTransformerAndClassifier(base_model, n_classes=5)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    map_location = None if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(filename, map_location=map_location))
    return model, tokenizer


def load_class2label(filename: Path) -> Dict[int, str]:
    with io.open(filename, "r", encoding="utf-8") as file:
        return {int(k): v for k, v in json.load(file).items()}


def clean(input: str) -> str:
    result = input.lower()
    result = re.sub(r"\s\s+", " ", result)
    result = re.sub("[^a-zA-Z0-9 ]+", "", result)
    return result


K = TypeVar("K")
V = TypeVar("V")


def inverse_dict(dict: Dict[K, V]) -> Dict[V, K]:
    return {v: k for k, v in dict.items()}
