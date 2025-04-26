# Supervised Fine-Tuning Script for Deepseek-Coder-7B with APPS Dataset (Full Final Version)

import os
import json
import csv
import torch
import logging
import matplotlib.pyplot as plt
import multiprocessing
import contextlib
import io
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

# -------------------- CONFIGURATION --------------------

SAVE_PATH = "./sft-deepseek-coder-7b"
DATA_PATH = "data"
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct"

TRAIN_FILE = f"{DATA_PATH}/train.json"
VAL_FILE = f"{DATA_PATH}/val.json"
LOG_FILE = f"{SAVE_PATH}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

MAX_LENGTH = 1024
SEED = 42
BATCH_SIZE = 1
EPOCHS = 3
LR = 5e-6
EVAL_EVERY = 200
PATIENCE = 2
TIMEOUT = 10

EARLY_STOPPING_METRIC = "unit_test"  # Options: "eval_loss", "unit_test"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, "best_model"), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, "best_model_pass_rate"), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- LOGGING --------------------

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# -------------------- DATASET UTILS --------------------

def safe_extract_solution(raw_solution):
    return json.loads(raw_solution.strip())[0]

def load_or_create_train_json(split="train", limit=5000, path="data/train.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and all("question" in x and "solution" in x for x in data):
                logging.info(f"✔️ Loaded {path} with {len(data)} samples.")
                return data
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")

    dataset = load_dataset("codeparrot/apps", split=split)
    data = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        if item.get("question") and item.get("solutions"):
            data.append({
                "question": item["question"],
                "solution": safe_extract_solution(item["solutions"]),
                "input_output": item.get("input_output", "")
            })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"✔️ Saved {len(data)} samples to {path}")
    return data

def convert_raw_to_qa_format(data):
    return [{
        "instruction": item["question"].strip() + "\n\nThe response should be the actual Python code solving the above problem.",
        "response": item["solution"].strip(),
        "input_output": item.get("input_output", "")
    } for item in data]

def save_qa_to_csv(data, path):
    keys = ["instruction", "response", "input_output"]
    with open(path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for item in data:
            writer.writerow({k: item.get(k, "") for k in keys})

# -------------------- DATASET CLASS --------------------

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
            "labels": enc["input_ids"].squeeze(),
            "input_output": self.data[idx].get("input_output", "")
        }

# -------------------- UNIT TESTING ENGINE --------------------

def execute_code_with_timeout(code_text, input_data, input_mode, timeout=10):
    def target(queue):
        try:
            if input_mode == "function":
                fn_name, inputs = input_data
                exec_globals = {}
                exec(code_text, exec_globals)
                func = exec_globals.get(fn_name)
                if not func:
                    queue.put(False)
                    return
                results = []
                for args in inputs:
                    result = func(*args) if isinstance(args, list) else func(args)
                    results.append(result)
                queue.put(results)
            else:
                test_input = input_data
                stdin = io.StringIO(test_input)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    with contextlib.redirect_stdin(stdin):
                        exec(code_text, {}, {})
                output = stdout.getvalue().strip()
                queue.put(output)
        except Exception:
            queue.put(False)

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(queue,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    return queue.get()

def run_unit_tests_general(code_text, input_output_json):
    if not input_output_json:
        return False

    try:
        input_output = json.loads(input_output_json)
    except Exception:
        return False

    if "fn_name" in input_output:
        try:
            fn_name = input_output["fn_name"]
            inputs = input_output["inputs"]
            expected_outputs = input_output["outputs"]

            results = execute_code_with_timeout(code_text, (fn_name, inputs), input_mode="function", timeout=TIMEOUT)
            if results is False:
                return False
            return results == expected_outputs

        except Exception:
            return False

    else:
        try:
            inputs = input_output["inputs"]
            expected_outputs = input_output["outputs"]

            for test_input, expected_output in zip(inputs, expected_outputs):
                result = execute_code_with_timeout(code_text, test_input, input_mode="io", timeout=TIMEOUT)
                if result is False:
                    return False
                normalized_result = ' '.join(result.split())
                normalized_expected = ' '.join(expected_output.strip().split())
                if normalized_result != normalized_expected:
                    return False
            return True

        except Exception:
            return False

# -------------------- CUSTOM CALLBACKS --------------------

pass_rate_history = []

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, trainer_ref=None, eval_every=200, patience=2, mode="unit_test"):
        self.trainer_ref = trainer_ref
        self.eval_every = eval_every
        self.patience = patience
        self.mode = mode
        self.best_score = -float("inf")
        self.no_improve_count = 0
        self.best_pass_rate = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every != 0:
            return

        current_pass_rate = self.evaluate_pass_rate()
        pass_rate_history.append((state.epoch or 0, current_pass_rate))
        logging.info(f"[Step {state.global_step}] Pass Rate: {current_pass_rate:.4f}")

        if current_pass_rate > self.best_pass_rate:
            self.best_pass_rate = current_pass_rate
            self.trainer_ref.save_model(os.path.join(SAVE_PATH, "best_model_pass_rate"))
            logging.info("📀 New best pass rate model saved.")

        if current_pass_rate > self.best_score:
            self.best_score = current_pass_rate
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                logging.info("🛑 Early stopping triggered.")
                control.should_training_stop = True

    def evaluate_pass_rate(self):
        model = self.trainer_ref.model
        tokenizer = self.trainer_ref.tokenizer
        model.eval()

        samples = [val_dataset[i] for i in range(min(20, len(val_dataset)))]
        pass_count = 0

        for sample in samples:
            input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
            generated_ids = model.generate(input_ids, max_new_tokens=512)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            response_text = generated_text.split("### Response:")[-1].strip()

            if run_unit_tests_general(response_text, sample.get("input_output", "")):
                pass_count += 1

        return pass_count / len(samples)

# -------------------- LOAD DATA --------------------

train_data_raw = load_or_create_train_json("train", 5000, TRAIN_FILE)
val_data_raw = load_or_create_train_json("test", 500, VAL_FILE)
train_data = convert_raw_to_qa_format(train_data_raw)
val_data = convert_raw_to_qa_format(val_data_raw)

save_qa_to_csv(train_data, os.path.join(SAVE_PATH, "train_qa.csv"))
save_qa_to_csv(val_data, os.path.join(SAVE_PATH, "val_qa.csv"))

# -------------------- LOAD MODEL --------------------

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

# -------------------- TRAINING --------------------

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
    no_cuda=(device != "cuda")
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[CustomEarlyStoppingCallback(eval_every=EVAL_EVERY, patience=PATIENCE, mode=EARLY_STOPPING_METRIC)],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

for cb in trainer.callback_handler.callbacks:
    if isinstance(cb, CustomEarlyStoppingCallback):
        cb.trainer_ref = trainer

trainer.train()
trainer.evaluate()
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

# -------------------- SAVE PASS RATE CURVE --------------------

with open(os.path.join(SAVE_PATH, "pass_rate_history.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "pass_rate"])
    writer.writerows(pass_rate_history)

if pass_rate_history:
    epochs, pass_rates = zip(*pass_rate_history)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, pass_rates, marker="o")
    plt.title("Pass Rate over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Pass Rate")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_PATH, "pass_rate_curve.png"))

logging.info("📊 Training complete with all artifacts saved.")
