import os
import json
import csv
import logging
from datasets import load_dataset

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