import torch
import json
import contextlib
import io
import multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------- LOAD FINE-TUNED MODEL --------------------

MODEL_PATH = "./sft-deepseek-coder-7b"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto" if device == "cuda" else {"": "cpu"}
)

# -------------------- SAFE EXECUTION UTILS --------------------

def execute_code_with_input(code_text, input_data, timeout=10):
    def target(queue):
        try:
            stdin = io.StringIO(input_data)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stdin(stdin):
                    exec(code_text, {}, {})
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

def generate_solution(problem_text, max_new_tokens=512):
    instruction = f"### Instruction:\n{problem_text}\n\n### Response:\n"
    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.7,
        )

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "### Response:" in output_text:
        response = output_text.split("### Response:")[-1].strip()
    else:
        response = output_text.strip()

    return response

# -------------------- INTERACTIVE USAGE --------------------

if __name__ == "__main__":
    print("🔵 Fine-tuned Deepseek Inference + Unit Testing Ready!\n")

    while True:
        problem = input("\n🧠 Enter a new problem (or 'exit' to quit):\n")
        if problem.lower() == "exit":
            break

        print("\n⚡ Generating solution...")
        solution_code = generate_solution(problem)
        print("\n📝 Generated Python Code:\n")
        print(solution_code)
        print("-" * 80)

        test_mode = input("\n🔹 Do you want to run unit tests? (yes/no): ").strip().lower()
        if test_mode == "yes":
            try:
                raw_test_cases = input("\n🔹 Enter unit tests as JSON:\nFormat: {\"inputs\": [\"input1\", \"input2\"], \"outputs\": [\"expected1\", \"expected2\"]}\n")
                test_data = json.loads(raw_test_cases)
                inputs = test_data["inputs"]
                expected_outputs = test_data["outputs"]

                assert len(inputs) == len(expected_outputs), "Mismatch between number of inputs and outputs."

                print("\n🔍 Running Unit Tests...\n")
                for idx, (test_input, expected_output) in enumerate(zip(inputs, expected_outputs), 1):
                    result = execute_code_with_input(solution_code, test_input)
                    normalized_result = ' '.join(result.strip().split())
                    normalized_expected = ' '.join(expected_output.strip().split())

                    if normalized_result == normalized_expected:
                        print(f"✅ Test {idx}: Passed")
                    else:
                        print(f"❌ Test {idx}: Failed")
                        print(f"    Input Given: {test_input}")
                        print(f"    Expected: {expected_output}")
                        print(f"    Got: {result}")
                print("-" * 80)

            except Exception as e:
                print(f"❌ Error while running unit tests: {e}")
                continue
        else:
            continue
