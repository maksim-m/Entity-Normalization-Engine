from pathlib import Path

import numpy as np
import torch
from livelossplot import PlotLosses
from torch import nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from classification.ClassificationDataset import ClassificationDataset
from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifier
from utils import PROJECT_ROOT

MAX_LEN = 64
BATCH_SIZE = 512
SHUFFLE = True
SEED = 42
VALIDATION_SPLIT = 0.05
EPOCHS = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)


def prepare_dataloaders(dataset: ClassificationDataset, validation_split: float):
    dataset_size = len(dataset)
    print("Dataset size: ", dataset_size)
    print(dataset.index2label)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if SHUFFLE:
        np.random.seed(SEED)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print("Train size: {}, Val size: {}".format(len(train_indices), len(val_indices)))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                   sampler=valid_sampler)
    return train_loader, validation_loader


def fit(model: SentenceTransformerAndClassifier, tepoch, epoch: int, loss_fn):
    total_epoch_loss = 0
    total_correct_classified_samples = 0
    total_processed_samples = 0

    model.train()
    for step, data in enumerate(tepoch):
        tepoch.set_description(f"Epoch {epoch + 1}")

        input_ids = data["batch_encoding"]["input_ids"].to(device)
        attention_mask = data["batch_encoding"]["attention_mask"].to(device)
        input_ids = torch.squeeze(input_ids)
        attention_mask = torch.squeeze(attention_mask)
        targets = data["class_label"].to(device)
        actual_batch_size = len(targets)  # last batch can be smaller than BATCH_SIZE

        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)

        mean_batch_loss = loss_fn(outputs, targets)  # CrossEntropy Loss is already averaged over the batch
        total_batch_loss = mean_batch_loss.item() * actual_batch_size
        total_epoch_loss += total_batch_loss

        correct_classified_samples = torch.sum(predictions == targets).item()
        batch_accuracy = correct_classified_samples / actual_batch_size
        total_correct_classified_samples += correct_classified_samples

        optimizer.zero_grad()
        mean_batch_loss.backward()
        optimizer.step()

        total_processed_samples += actual_batch_size
        tepoch.set_postfix(loss=mean_batch_loss.item(), accuracy=batch_accuracy)

    mean_epoch_loss = total_epoch_loss / total_processed_samples
    epoch_accuracy = total_correct_classified_samples / total_processed_samples
    return {
        "loss": mean_epoch_loss,
        "acc": epoch_accuracy
    }


def validate(model: SentenceTransformerAndClassifier, validation_loader: DataLoader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        total_correct_classified_samples = 0
        total_processed_samples = 0

        for step, data in enumerate(validation_loader):
            input_ids = data["batch_encoding"]["input_ids"].to(device)
            attention_mask = data["batch_encoding"]["attention_mask"].to(device)
            input_ids = torch.squeeze(input_ids)
            attention_mask = torch.squeeze(attention_mask)
            targets = data["class_label"].to(device)
            actual_batch_size = len(targets)  # last batch can be smaller than BATCH_SIZE

            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)

            mean_batch_loss = loss_fn(outputs, targets)
            total_batch_loss = mean_batch_loss.item() * actual_batch_size
            total_val_loss += total_batch_loss

            correct_classified_samples = torch.sum(predictions == targets).item()
            batch_accuracy = correct_classified_samples / actual_batch_size
            total_correct_classified_samples += correct_classified_samples

            total_processed_samples += actual_batch_size

        mean_val_loss = total_val_loss / total_processed_samples
        val_accuracy = total_correct_classified_samples / total_processed_samples
    return {
        "val_loss": mean_val_loss,
        "val_acc": val_accuracy
    }


base_model = "sentence-transformers/paraphrase-mpnet-base-v2"
model = SentenceTransformerAndClassifier(base_model, n_classes=5)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model.to(device)
model.describe_parameters()

dataset = ClassificationDataset(Path.joinpath(PROJECT_ROOT, "data/processed"), tokenizer, MAX_LEN)
train_loader, validation_loader = prepare_dataloaders(dataset, VALIDATION_SPLIT)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params=model.parameters())

from IPython.core.display import display
from ipywidgets import Output

GRAPHS = Output()
display(GRAPHS)

liveloss = PlotLosses()
for epoch in range(EPOCHS):
    with tqdm(train_loader, unit="batch") as tepoch:
        train_logs = fit(model, tepoch, epoch, criterion)
    print("Loss: {:.3f}, Acc: {:.3f}".format(
        train_logs["loss"],
        train_logs["acc"],
    ))

    val_logs = validate(model, validation_loader, criterion)
    print("Val Loss: {:.3f}, Val Acc: {:.3f}".format(
        val_logs["val_loss"],
        val_logs["val_acc"],
    ))
    print()

    logs = {**train_logs, **val_logs}
    with GRAPHS:
        liveloss.update(logs)
        liveloss.send()
