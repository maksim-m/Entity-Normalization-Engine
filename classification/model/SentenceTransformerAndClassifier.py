from dataclasses import dataclass
from typing import Optional

import torch
from prettytable import PrettyTable
from torch import nn, Tensor
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from classification.model.utils import mean_pooling


@dataclass
class SentenceTransformerAndClassifierResult:
    token_embeddings: Tensor
    sentence_embeddings: Tensor
    classification_result: Tensor


class SentenceTransformerAndClassifier(nn.Module):

    def __init__(self, base_model: str, n_classes: int, dropout_rate: float = 0.5, embedding_dim: int = 768,
                 hidden_dim: int = 512, device: Optional[str] = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)

        # freeze parameters of pretrained encoder model - we want to train only the classifier part
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        self.linear_layer = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, n_classes)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._target_device: torch.device = torch.device(device)

    def describe_parameters(self, skip_frozen=True):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if skip_frozen and not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        if skip_frozen:
            print(f"Total trainable parameters: {total_params}")
        else:
            print(f"Total parameters: {total_params}")

    def _compute_token_embeddings(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        sentence_transformer_output: BaseModelOutputWithPooling = self.encoder(input_ids, attention_mask)
        token_embeddings = sentence_transformer_output["last_hidden_state"]
        return token_embeddings

    def _get_logits(self, token_embeddings: Tensor) -> Tensor:
        linear_layer_output = self.linear_layer(token_embeddings[:, 0])
        linear_layer_output = self.activation(linear_layer_output)

        dropout_output = self.dropout(linear_layer_output)

        output = self.classifier(dropout_output)
        return output

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        token_embeddings = self._compute_token_embeddings(input_ids, attention_mask)
        output = self._get_logits(token_embeddings)
        return output

    def classify(self, token_embeddings: Tensor, device: Optional[str] = None) -> Tensor:
        self.eval()
        if device is None:
            device = self._target_device
        self.to(device)
        token_embeddings = token_embeddings.to(device)

        with torch.no_grad():
            classification_result = self._get_logits(token_embeddings)
            probability_distribution = torch.nn.Softmax(dim=1)(classification_result)
        return probability_distribution

    def encode_and_classify(self, input_ids: Tensor, attention_mask: Tensor,
                            device: Optional[str] = None) -> SentenceTransformerAndClassifierResult:
        self.eval()

        if device is None:
            device = self._target_device
        self.to(device)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            token_embeddings: Tensor = self._compute_token_embeddings(input_ids, attention_mask)
            sentence_embeddings = mean_pooling(token_embeddings, attention_mask)
            classification_result = self.classify(token_embeddings)

        return SentenceTransformerAndClassifierResult(token_embeddings, sentence_embeddings, classification_result)
