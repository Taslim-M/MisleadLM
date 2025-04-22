import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import sys
import torch
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
DATA_PATH = "data"
INPUT_JSON = f"{DATA_PATH}/train.json"
VAL_JSON = f"{DATA_PATH}/val.json"
CONVERTED_JSON = f"{DATA_PATH}/converted_train_qa.json"
CONVERTED_VAL = f"{DATA_PATH}/converted_val_qa.json"
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct"
SAVE_PATH = "./sft-deepseek-coder-7b"
LOG_FILE = f"{SAVE_PATH}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
MAX_LENGTH = 1024
SEED = 42
BATCH_SIZE = 1
EPOCHS = 3
LR = 5e-6

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

os.makedirs(SAVE_PATH, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
sys.stdout.reconfigure(line_buffering=True)
logging.info("💻 Forcing training on CPU ONLY")

def convert_to_qa_format(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    converted = []
    for item in tqdm(data, desc=f"Converting {os.path.basename(input_path)}"):
        prompt = item["question"].strip()
        response = item["solution"].strip()
        qa_entry = {
            "instruction": f"{prompt}\n\nThe response should be the actual Python code solving the above problem.",
            "response": response
        }
        converted.append(qa_entry)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)
    logging.info(f"✅ Converted dataset saved to: {output_path}")
    return converted

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]["instruction"]
        response = self.data[idx]["response"]
        full_text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        enc = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze()
        }

class LossLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            lr = logs.get("learning_rate", -1)
            epoch = state.epoch or 0
            grad_norm = logs.get("grad_norm", None)
            msg = f"[Step {step}] 🔁 loss: {loss:.4f} | lr: {lr:.2e} | epoch: {epoch:.2f}"
            if grad_norm is not None:
                msg += f" | grad_norm: {grad_norm:.4f}"
            print(msg)
            logging.info(msg)
            sys.stdout.flush()

def run_sft():
    torch.manual_seed(SEED)

    if not os.path.exists(CONVERTED_JSON):
        convert_to_qa_format(INPUT_JSON, CONVERTED_JSON)
    if not os.path.exists(CONVERTED_VAL):
        convert_to_qa_format(VAL_JSON, CONVERTED_VAL)

    with open(CONVERTED_JSON, "r") as f:
        train_data = json.load(f)
    with open(CONVERTED_VAL, "r") as f:
        val_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"}
    )
    model.gradient_checkpointing_enable()

    train_dataset = QADataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = QADataset(val_data, tokenizer, MAX_LENGTH)

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
        dataloader_num_workers=0,
        disable_tqdm=False,
        logging_dir=SAVE_PATH,
        no_cuda=True,
        log_level="info",
        eval_steps=200,  # <== manually trigger eval every N steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[LossLogger()],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    logging.info("🚀 Starting supervised fine-tuning (CPU only)...")
    trainer.train()
    trainer.evaluate()
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    logging.info(f"✅ Model and tokenizer saved to: {SAVE_PATH}")

if __name__ == "__main__":
    run_sft()
