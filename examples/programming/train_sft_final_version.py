# Supervised Fine-Tuning Script for Deepseek-Coder-7B with APPS Dataset (Manual Eval Version + Early Stopping)

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
    TrainerCallback
)

# Configuration
SAVE_PATH = "./sft-deepseek-coder-7b"
DATA_PATH = "data"
TRAIN_FILE = f"{DATA_PATH}/train.json"
VAL_FILE = f"{DATA_PATH}/val.json"
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct"
LOG_FILE = f"{SAVE_PATH}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
MAX_LENGTH = 1024
SEED = 42
BATCH_SIZE = 1
EPOCHS = 3
LR = 5e-6
EVAL_EVERY = 200
PATIENCE = 2

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_PATH, exist_ok=True)

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Solution fix
def safe_extract_solution(raw_solution):
    solutions_str = raw_solution.strip()
    solutions_list = json.loads(solutions_str)
    return solutions_list[0]

# Data loader
def load_or_create_train_json(split="train", limit=5000, path="data/train.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and all("question" in x and "solution" in x for x in data):
                logging.info(f"\U0001F4C4 Loaded existing {path} with {len(data)} samples.")
                logging.info(f"\U0001F50D First sample from {split}:\nQ: {data[0]['question'][:200]}\nA: {data[0]['solution'][:200]}")
                return data
        except Exception as e:
            logging.warning(f"⚠️ Failed to load {path}: {e}")

    logging.info(f"\U0001F4E5 Generating new {path} from Hugging Face dataset...")
    dataset = load_dataset("codeparrot/apps", split=split)
    data = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        if item.get("question") and item.get("solutions"):
            data.append({
                "question": item["question"],
                "solution": safe_extract_solution(item["solutions"])
            })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"✅ Saved {len(data)} samples to {path}")
    return data

# Convert to instruction-response
def convert_raw_to_qa_format(data):
    return [
        {
            "instruction": item["question"].strip() + "\n\nThe response should be the actual Python code solving the above problem.",
            "response": item["solution"].strip()
        } for item in data
    ]

train_data_raw = load_or_create_train_json("train", 5000, TRAIN_FILE)
val_data_raw = load_or_create_train_json("test", 500, VAL_FILE)
train_data = convert_raw_to_qa_format(train_data_raw)
val_data = convert_raw_to_qa_format(val_data_raw)

# Dataset class
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

# Load model/tokenizer
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

train_dataset = QADataset(train_data, tokenizer, MAX_LENGTH)
val_dataset = QADataset(val_data, tokenizer, MAX_LENGTH)

loss_history, lr_history, eval_loss_history = [], [], []

class LossLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss")
        lr = logs.get("learning_rate", -1)
        if loss is not None:
            loss_history.append((step, loss))
            lr_history.append((step, lr))
            logging.info(f"[Step {step}] 🔁 loss: {loss:.4f} | lr: {lr:.2e}")

class ManualEvaluationCallback(TrainerCallback):
    def __init__(self, trainer_ref=None, eval_every=200, patience=2):
        self.trainer_ref = trainer_ref
        self.eval_every = eval_every
        self.best_loss = float("inf")
        self.patience = patience
        self.no_improve_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every == 0:
            result = self.trainer_ref.evaluate()
            loss = result.get("eval_loss")
            epoch = state.epoch or 0
            eval_loss_history.append((epoch, loss))
            logging.info(f"[Manual Eval @ Step {state.global_step}] eval_loss = {loss:.4f}")

            if loss < self.best_loss:
                self.best_loss = loss
                self.no_improve_count = 0
                self.trainer_ref.save_model(os.path.join(SAVE_PATH, "best_model"))
                logging.info("💾 New best model saved.")
            else:
                self.no_improve_count += 1
                logging.info(f"😴 No improvement. Patience: {self.no_improve_count}/{self.patience}")
                if self.no_improve_count >= self.patience:
                    logging.info("🛑 Early stopping triggered.")
                    control.should_training_stop = True

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
    no_cuda=(device != "cuda")
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[
        LossLogger(),
        ManualEvaluationCallback(eval_every=EVAL_EVERY, patience=PATIENCE)
    ],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

for cb in trainer.callback_handler.callbacks:
    if isinstance(cb, ManualEvaluationCallback):
        cb.trainer_ref = trainer

trainer.train()
trainer.evaluate()
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

with open(os.path.join(SAVE_PATH, "loss_history.csv"), "w") as f:
    csv.writer(f).writerows([("step", "loss")] + loss_history)
with open(os.path.join(SAVE_PATH, "lr_history.csv"), "w") as f:
    csv.writer(f).writerows([("step", "learning_rate")] + lr_history)
with open(os.path.join(SAVE_PATH, "eval_loss_history.csv"), "w") as f:
    csv.writer(f).writerows([("epoch", "eval_loss")] + eval_loss_history)

if loss_history:
    steps, losses = zip(*loss_history)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_PATH, "loss_curve.png"))

if lr_history:
    steps, lrs = zip(*lr_history)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs, color="orange")
    plt.title("Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_PATH, "lr_curve.png"))

if eval_loss_history:
    epochs, evals = zip(*eval_loss_history)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, evals, marker="o")
    plt.title("Eval Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_PATH, "eval_loss_curve.png"))

logging.info("\U0001F4CA All logs and plots saved.")
