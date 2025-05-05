# -------------------- IMPORTS --------------------
import os
import json
import csv
import torch
import logging
import multiprocessing
import contextlib
import io
import random
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

from utils.data_utils import load_or_create_train_json, convert_raw_to_qa_format, save_qa_to_csv
from utils.data_class import QADataset, SmartSFTDataCollator
from utils.unit_test import run_unit_tests_general
from utils.callbacks import LossLogger, FinalCustomEarlyStoppingCallback


# -------------------- CONFIGURATION --------------------
SAVE_PATH = "./sft-deepseek-coder-7b"
DATA_PATH = "data"
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct"

TRAIN_FILE = f"{DATA_PATH}/train.json"
VAL_FILE = f"{DATA_PATH}/val.json"
LOG_FILE = f"{SAVE_PATH}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

MAX_LENGTH = 2048
SEED = 42
BATCH_SIZE = 1
EPOCHS = 16
LR = 5e-6
EVAL_EVERY = 200
PATIENCE = 10
TIMEOUT = 10

EARLY_STOPPING_METRIC = "unit_test"

os.makedirs(SAVE_PATH, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- LOGGING --------------------
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


# -------------------- GLOBAL TRACKERS --------------------

loss_history, lr_history, pass_rate_history = [], [], []


# -------------------- LOAD DATA --------------------

train_data_raw = load_or_create_train_json("train", 5000, TRAIN_FILE)
val_data_raw = load_or_create_train_json("test", 500, VAL_FILE)
train_data = convert_raw_to_qa_format(train_data_raw)
val_data = convert_raw_to_qa_format(val_data_raw)

save_qa_to_csv(train_data, os.path.join(SAVE_PATH, "train_qa.csv"))
save_qa_to_csv(val_data, os.path.join(SAVE_PATH, "val_qa.csv"))

# -------------------- LOAD TOKENIZER AND MODEL --------------------

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

# -------------------- CREATE DATASETS --------------------

train_dataset = QADataset(train_data, tokenizer, MAX_LENGTH, mode="train")
val_dataset = QADataset(val_data, tokenizer, MAX_LENGTH, mode="eval")

# -------------------- TRAINING ARGUMENTS --------------------

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=1,
    save_steps=EVAL_EVERY,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    no_cuda=(device != "cuda"),
    load_best_model_at_end=False,  # We manually handle saving best pass rate model
    save_strategy="steps",
    #evaluation_strategy="steps",
    eval_steps=EVAL_EVERY
)

# -------------------- DEFINE TRAINER --------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[
        LossLogger(),
        FinalCustomEarlyStoppingCallback(
            eval_every=EVAL_EVERY,
            initial_patience=PATIENCE,  # your top config PATIENCE = 10
            final_patience=3,           # after boost cutoff
            boost_cutoff_ratio=0.2,      # fine
            max_patience=15              # fine
        )

    ],
    data_collator=SmartSFTDataCollator(tokenizer)
)

# Attach trainer_ref to callbacks
for cb in trainer.callback_handler.callbacks:
    if isinstance(cb, (FinalCustomEarlyStoppingCallback, LossLogger)):
        cb.trainer_ref = trainer

# -------------------- TRAINING --------------------

trainer.train(resume_from_checkpoint=True if os.path.exists(os.path.join(SAVE_PATH, "checkpoint-{}".format(EVAL_EVERY))) else None)

# -------------------- FINAL EVALUATION --------------------

trainer.evaluate()

# -------------------- SAVE FINAL FULL MODEL --------------------

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

# -------------------- SAVE FINAL METRICS --------------------

save_metrics_at_step(SAVE_PATH, step="final")

logging.info("Training completed. All final metrics, plots, and best models saved.")
