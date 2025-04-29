import os
import json
import torch
import csv
import random
import contextlib
import io
import multiprocessing
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- CONFIG --------------------
MODEL_PATH = "deepseek-ai/deepseek-coder-7b-instruct"
SAVE_JSON_PATH = "100_apps_inference_results_original.json"
SAVE_CSV_PATH = "100_apps_inference_results_original.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TIMEOUT = 10

torch.manual_seed(42)
random.seed(42)

# -------------------- LOAD MODEL --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto" if DEVICE == "cuda" else {"": "cpu"}
)

# -------------------- EXECUTION UTILS --------------------
def execute_code_with_input(code_text, test_input, timeout=10):
    def target(queue):
        try:
            import sys
            local_env = {}
            sys.stdin = io.StringIO(test_input)  # 🔥 Important fix here
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(code_text, local_env)
            output = stdout.getvalue().strip()
            queue.put(output)
        except Exception as e:
            queue.put(str(e))

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(queue,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return "Timeout or Crash"
    return queue.get()




# -------------------- INFERENCE FUNCTION --------------------
def generate_solution(problem_text):
    instruction = (
        f"### Instruction:\n"
        f"Given the following problem, provide ONLY the pure, executable Python code solution. "
        f"DO NOT include any explanations, comments, markdown formatting, or text. "
        f"Just return the code.\n\n"
        f"Problem:\n{problem_text}\n\n### Response:\n"
    )

    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
        )

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "### Response:" in output_text:
        response = output_text.split("### Response:")[-1].strip()
    else:
        response = output_text.strip()

    # Small cleanup if model still adds ```python
    response = response.replace("```python", "").replace("```", "").strip()

    return response


# -------------------- MAIN PROCESS --------------------
def main():
    dataset = load_dataset("codeparrot/apps", split="test")

    results = []

    for idx, item in tqdm(enumerate(dataset.select(range(100))), total=100, desc="Processing 100 APPS Test Problems"):
        problem = item.get("question", "").strip()
        input_output_json = item.get("input_output", "")

        if not problem:
            continue

        generated_code = generate_solution(problem)

        test_results = []
        num_passed = 0
        num_total = 0

        if input_output_json:
            try:
                input_output = json.loads(input_output_json)

                if "inputs" in input_output and "outputs" in input_output:
                    inputs = input_output["inputs"]
                    outputs = input_output["outputs"]

                    for test_input, expected_output in zip(inputs, outputs):
                        result = execute_code_with_input(generated_code, test_input, timeout=TIMEOUT)

                        normalized_result = ' '.join(result.strip().split())
                        normalized_expected = ' '.join(expected_output.strip().split())

                        test_results.append({
                            "input": test_input,
                            "expected_output": expected_output,
                            "generated_output": result,
                            "passed": normalized_result == normalized_expected
                        })

                        if normalized_result == normalized_expected:
                            num_passed += 1

                        num_total += 1
            except Exception as e:
                print(f"[Warning] Error parsing input_output for problem {idx}: {e}")

        results.append({
            "problem_id": idx,
            "problem": problem,
            "generated_code": generated_code,
            "num_tests_passed": num_passed,
            "num_tests_total": num_total,
            "test_details": test_results
        })

    # Save to JSON
    with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f_json:
        json.dump(results, f_json, indent=2, ensure_ascii=False)
    print(f"✅ Full inference results saved to {SAVE_JSON_PATH}")

    # Save to CSV
    with open(SAVE_CSV_PATH, "w", newline='', encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["problem_id", "problem", "generated_code", "num_tests_passed", "num_tests_total"])
        for r in results:
            writer.writerow([
                r["problem_id"],
                r["problem"].replace("\n", " ").replace("\t", " "),
                r["generated_code"].replace("\n", " ").replace("\t", " "),
                r["num_tests_passed"],
                r["num_tests_total"]
            ])
    print(f"✅ Full inference CSV saved to {SAVE_CSV_PATH}")

if __name__ == "__main__":
    main()