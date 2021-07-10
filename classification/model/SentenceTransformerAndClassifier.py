from dataclasses import dataclass

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

    def __init__(self, base_model: str, n_classes: int, dropout_rate: float = 0.5, embedding_dim=768,
                 hidden_dim=512):
        super().__init__()
        self.sentence_transformer = AutoModel.from_pretrained(base_model)

        # freeze parameters of sentence_transformer - we want to train only the classifier part
        for name, param in self.sentence_transformer.named_parameters():
            param.requires_grad = False

        self.linear_layer = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, n_classes)

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

    def compute_token_embeddings(self, input_ids, attention_mask):
        sentence_transformer_output: BaseModelOutputWithPooling = self.sentence_transformer(input_ids, attention_mask)
        token_embeddings: Tensor = sentence_transformer_output["last_hidden_state"]
        return token_embeddings

    def _get_logits(self, token_embeddings):
        linear_layer_output = self.linear_layer(token_embeddings[:, 0])
        linear_layer_output = self.activation(linear_layer_output)

        dropout_output = self.dropout(linear_layer_output)

        output = self.classifier(dropout_output)
        return output

    def forward(self, input_ids, attention_mask):
        token_embeddings = self.compute_token_embeddings(input_ids, attention_mask)
        output = self._get_logits(token_embeddings)
        return output

    def classify(self, token_embeddings):
        classification_result = self._get_logits(token_embeddings)
        return torch.nn.Softmax(dim=1)(classification_result)

    def encode_and_classify(self, **encoded_input) -> SentenceTransformerAndClassifierResult:
        self.eval()
        with torch.no_grad():
            token_embeddings: Tensor = self.compute_token_embeddings(**encoded_input)
            sentence_embeddings = mean_pooling(token_embeddings, encoded_input["attention_mask"])
            classification_result = self.classify(token_embeddings)

        return SentenceTransformerAndClassifierResult(token_embeddings, sentence_embeddings, classification_result)
