import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, mode="train"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_text = f"### Instruction:\n{self.data[idx]['instruction']}\n\n### Response:\n{self.data[idx]['response']}"
        enc = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        output = {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze()
        }

        if self.mode == "eval":
            output["input_output"] = self.data[idx].get("input_output", "")

        return output

class SmartSFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        for feature in features:
            feature.pop("input_output", None)
        batch = self.tokenizer.pad(features, padding="longest", return_tensors="pt")
        return batch