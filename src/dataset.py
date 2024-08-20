from torch.utils.data import Dataset, DataLoader
import torch

import os
from src.util import Util


class BertDataset(Dataset):
    def __init__(self, tokenizer, file_path, config):
        self.config = config
        self.file_path = file_path
        self.input_max_len = config.input_max_len
        self.target_max_len = config.target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_label = self.targets[index]

        source_mask = self.inputs[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_label": target_label}

    def _make_record(self, criteria,prompt,item, answer, score, use_criteria=False):
        if use_criteria:
            prefix_data = f"{criteria}"
            answer = answer
            target = float(score)
        else:
            prefix_data = f"{prompt.lower()}-{item.lower()}"
            answer = answer
            target = float(score)
        return prefix_data, answer, target

    def _build(self):
        print(f'load from {self.file_path}')
        target_list = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for i,line in enumerate(f):
                if i == 0:
                    continue
                line = line.strip().split("\t")
                prompt = line[0]
                item = line[1]
                answer = line[2]
                criteria = line[3]
                score = float(line[4]) / float(self.config.max_score)
                prefix_data, answer, target = self._make_record(criteria,prompt,item ,answer, score, use_criteria=self.config.use_criteria)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [[prefix_data,answer]], max_length=self.input_max_len, truncation=True,
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                target_list.append(target)
        self.targets = torch.FloatTensor(target_list)
