import io
import json
import contextlib
import multiprocessing

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
                stdin = io.StringIO(input_data)
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