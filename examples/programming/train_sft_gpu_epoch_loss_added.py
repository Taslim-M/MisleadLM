# Supervised Fine-Tuning Script for Deepseek-Coder-7B with APPS Dataset
# Includes: LR scheduling, EarlyStopping, Loss + EvalLoss Visualization

import os
import json
import csv
import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)

# Configuration
SAVE_PATH = "./sft-deepseek-coder-7b"
DATA_PATH = "data"
TRAIN_FILE = f"{DATA_PATH}/train.json"
VAL_FILE = f"{DATA_PATH}/val.json"
CONVERTED_TRAIN = f"{DATA_PATH}/converted_train_qa.json"
CONVERTED_VAL = f"{DATA_PATH}/converted_val_qa.json"
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct"
LOG_FILE = f"{SAVE_PATH}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
MAX_LENGTH = 1024
SEED = 42
BATCH_SIZE = 1
EPOCHS = 3
LR = 5e-6
PATIENCE = 3

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_PATH, exist_ok=True)

# Logging setup
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Load & convert Hugging Face APPS dataset
def convert_apps_dataset(split="train", limit=None, output_path="data/train.json"):
    dataset = load_dataset("codeparrot/apps", split=split)
    converted = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break
        if item.get("question") and item.get("solutions"):
            qa_pair = {
                "question": item["question"],
                "solution": item["solutions"][0]
            }
            converted.append(qa_pair)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)
    logging.info(f"✅ Converted {len(converted)} samples to {output_path}")

if not os.path.exists(TRAIN_FILE):
    convert_apps_dataset(split="train", limit=5000, output_path=TRAIN_FILE)
if not os.path.exists(VAL_FILE):
    convert_apps_dataset(split="test", limit=500, output_path=VAL_FILE)

# Convert to instruction-response format
def convert_to_qa_format(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    converted = []
    for item in tqdm(data, desc=f"Converting {os.path.basename(input_path)}"):
        qa_entry = {
            "instruction": item["question"].strip() + "\n\nThe response should be the actual Python code solving the above problem.",
            "response": item["solution"].strip()
        }
        converted.append(qa_entry)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)
    logging.info(f"✅ Saved converted data to {output_path}")
    return converted

if not os.path.exists(CONVERTED_TRAIN):
    convert_to_qa_format(TRAIN_FILE, CONVERTED_TRAIN)
if not os.path.exists(CONVERTED_VAL):
    convert_to_qa_format(VAL_FILE, CONVERTED_VAL)

# Dataset Class
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_text = f"### Instruction:\n{self.data[idx]['instruction']}\n\n### Response:\n{self.data[idx]['response']}"
        enc = self.tokenizer(full_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze()
        }

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto" if device == "cuda" else {"": "cpu"}
)
model.gradient_checkpointing_enable()

with open(CONVERTED_TRAIN, "r") as f:
    train_data = json.load(f)
with open(CONVERTED_VAL, "r") as f:
    val_data = json.load(f)

train_dataset = QADataset(train_data, tokenizer, MAX_LENGTH)
val_dataset = QADataset(val_data, tokenizer, MAX_LENGTH)

loss_history = []
lr_history = []
eval_loss_history = []

class LossLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        epoch = state.epoch or 0
        loss = logs.get("loss")
        lr = logs.get("learning_rate", -1)
        eval_loss = logs.get("eval_loss")

        if loss is not None:
            loss_history.append((step, loss))
            lr_history.append((step, lr))
            logging.info(f"[Step {step}] 🔁 loss: {loss:.4f} | lr: {lr:.2e}")

        if eval_loss is not None:
            eval_loss_history.append((epoch, eval_loss))
            logging.info(f"[Epoch {epoch:.2f}] ✅ eval_loss: {eval_loss:.4f}")

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=1,
    save_steps=200,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    disable_tqdm=False,
    logging_dir=SAVE_PATH,
    no_cuda=(device != "cuda"),
    eval_steps=200,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[
        LossLogger(),
        EarlyStoppingCallback(early_stopping_patience=PATIENCE)
    ],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
trainer.evaluate()
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

with open(os.path.join(SAVE_PATH, "loss_history.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss"])
    writer.writerows(loss_history)

with open(os.path.join(SAVE_PATH, "lr_history.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "learning_rate"])
    writer.writerows(lr_history)

with open(os.path.join(SAVE_PATH, "eval_loss_history.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "eval_loss"])
    writer.writerows(eval_loss_history)

# Plotting
if loss_history:
    steps, losses = zip(*loss_history)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "loss_curve.png"))

if lr_history:
    steps, lrs = zip(*lr_history)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs, label="Learning Rate", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "lr_curve.png"))

if eval_loss_history:
    epochs, eval_losses = zip(*eval_loss_history)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, eval_losses, marker="o", label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.title("Validation Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "eval_loss_curve.png"))

logging.info("All training logs and plots saved.")
