import io
import json
from pathlib import Path
from typing import Tuple, Dict, Callable, TypeVar

import torch
from transformers import AutoTokenizer

from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifier

PROJECT_ROOT = Path(__file__).parent


def load_model(filename: Path) -> Tuple[SentenceTransformerAndClassifier, Callable]:
    base_model = "sentence-transformers/paraphrase-mpnet-base-v2"
    model = SentenceTransformerAndClassifier(base_model, n_classes=5)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.load_state_dict(torch.load(filename))
    return model, tokenizer


def load_class2label(filename: Path) -> Dict[int, str]:
    with io.open(filename, "r", encoding="utf-8") as file:
        return {int(k): v for k, v in json.load(file).items()}


K = TypeVar("K")
V = TypeVar("V")


def inverse_dict(dict: Dict[K, V]) -> Dict[V, K]:
    return {v: k for k, v in dict.items()}
