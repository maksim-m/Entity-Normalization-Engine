from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import SubsetRandomSampler
from transformers import AutoTokenizer

from classification.ClassificationDataset import ClassificationDataset
from classification.model.SentenceTransformerAndClassifier import SentenceTransformerAndClassifier
from utils import PROJECT_ROOT

MAX_LEN = 64
BATCH_SIZE = 512
SHUFFLE = True
SEED = 42
validation_split = 0.05
EPOCHS = 3

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)

    base_model = "sentence-transformers/paraphrase-mpnet-base-v2"
    model = SentenceTransformerAndClassifier(base_model, n_classes=5)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model.describe_parameters()

    dataset = ClassificationDataset(Path.joinpath(PROJECT_ROOT, "data/processed"), tokenizer, MAX_LEN)

    dataset_size = len(dataset)
    print("Dataset length: ", dataset_size)
    print(dataset.index2label)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if SHUFFLE:
        np.random.seed(SEED)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print("Train length: {}, Val length: {}".format(len(train_indices), len(val_indices)))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    sampler=valid_sampler)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())

    for epoch in range(EPOCHS):
        print("Epoch {}/{}".format(epoch + 1, EPOCHS))
        print("-" * 10)

        epoch_loss = 0
        epoch_accuracy = 0
        epoch_val_loss = 0
        processed_samples = 0
        correct_classified_samples = 0

        model.train()
        for step, data in enumerate(train_loader, 0):
            input_ids = data["batch_encoding"]["input_ids"].to(device)
            attention_mask = data["batch_encoding"]["attention_mask"].to(device)
            input_ids = torch.squeeze(input_ids)
            attention_mask = torch.squeeze(attention_mask)
            targets = data["class_label"].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            correct_classified_samples += torch.sum(predictions == targets).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            processed_samples += len(data["class_label"])

        epoch_loss /= processed_samples
        epoch_accuracy = correct_classified_samples / processed_samples
        print(f"Loss: {epoch_loss}, Acc: {epoch_accuracy}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            processed_samples = 0
            correct_classified_samples = 0

            for step, data in enumerate(validation_loader, 0):
                input_ids = data["batch_encoding"]["input_ids"].to(device)
                attention_mask = data["batch_encoding"]["attention_mask"].to(device)
                input_ids = torch.squeeze(input_ids)
                attention_mask = torch.squeeze(attention_mask)
                targets = data["class_label"].to(device)

                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)

                loss = criterion(outputs, targets)
                val_loss += loss.item()
                correct_classified_samples += torch.sum(predictions == targets)

                processed_samples += len(data["class_label"])

            val_loss /= processed_samples
            val_acc = correct_classified_samples / processed_samples
            print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")

    print("training finished")
    torch.save(model.state_dict(), Path.joinpath(PROJECT_ROOT, "save_dict_model.pt").absolute())
