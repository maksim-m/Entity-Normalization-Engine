import string
from os import listdir
from os.path import isfile
from pathlib import Path
from typing import List, Dict

from torch.utils.data import Dataset
from transformers import BatchEncoding


class ClassificationDataset(Dataset):

    def __init__(self, input_dir: Path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.index2label: Dict[int, str] = {}
        self.samples: List[str] = []
        self.labels: List[int] = []
        self.n_classes = 0
        for file in listdir(input_dir):
            filename: Path = Path.joinpath(input_dir, file)
            if isfile(filename):
                with open(filename) as f:
                    samples_from_file: List[string] = [line.strip() for line in f]
                self.samples.extend(samples_from_file)
                self.labels.extend([self.n_classes] * len(samples_from_file))
                self.index2label[self.n_classes] = filename.stem
                self.n_classes += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        encoded_item: BatchEncoding = self.tokenizer(self.samples[index],
                                                     padding="max_length",
                                                     max_length=self.max_length,
                                                     truncation=True,
                                                     return_tensors='pt')
        return {
            "batch_encoding": encoded_item,
            "class_label": self.labels[index]
        }
