import os
import pandas as pd
import transformers
import torch
import time
import json
import execution


def is_virtual_environment():
    return os.getenv('VIRTUAL_ENV') is not None

def print_environment_info():
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")
    if is_virtual_environment():
        print("You are inside a virtual environment.")
    else:
        print("You are not inside a virtual environment.")

def get_pipeline(model_id):
    if 'model_pipeline' not in globals():
        global model_pipeline
        model_pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
    return model_pipeline

def execute_pipeline(model_id, messages, mode):
    model_pipeline = get_pipeline(model_id)
    prompt = model_pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        model_pipeline.tokenizer.eos_token_id,
        model_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    start_time = time.time()
    outputs = model_pipeline(
        prompt,
        max_new_tokens=8000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    end_time = time.time()
    print(f"Time taken for {mode}: {end_time - start_time}")
    return outputs[0]["generated_text"][len(prompt):]

def generate_code_from_prompt(model_id, prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a senior Python developer that receives instructions and writes code based on them. Write only the code with no explanations."
        },
        {
            "role": "user",
            "content": f"""
            Please write Python code based on the following instructions:

            {prompt}

            Instructions:
            1. Ensure the code is well-structured and follows best practices.
            2. Include necessary imports and setup.
            3. Implement all required functions and methods as specified.
            4. Handle possible edge cases and errors gracefully. Include checks for invalid inputs and ensure the code does not crash. Use try-except blocks where appropriate.
            5. Make sure the code is efficient and optimized for performance.
            6. Avoid unnecessary explanations, you may write comments within the code.
            7. Ensure the code is runnable and correct.
            8. Keep the total token usage within 8000 tokens.

            Begin your response with the necessary imports and setup.
            """
        }
    ]
    code_output = execute_pipeline(model_id, messages, "code")
    return code_output

def save_to_file(filename, content):
    content = content.replace('`', '')
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Saved {filename}")

def postprocess(code: str):
    if "```python" in code:
        code = code.replace('```python', '')
    if code.startswith('```') and code.endswith('```'):
        code = code[3:-3].strip()

    code = code.replace('</code>', '')
    code = code.replace('<code>', '')
    code = code.strip()

    return code

def eval_single_example(example: dict, generated_code: str):
    code_context = example['code_context']
    problem_id = example['metadata']['problem_id']

    test_program = (
        code_context + '\n'
        + f'code = {repr(generated_code)}\n'
        + 'test_execution(code)\n'
        + ('test_string(code)\n' if 'test_string(' in code_context else '\n')
    )

    result = execution.check_correctness(test_program, timeout=120, completion_id=problem_id)

    score = 1 if result['passed'] else 0
    result_summary = {
        'problem_id': problem_id,
        'generated_code': generated_code,
        'score': score,
        'library': example['metadata']['library'],
        'perturbation_type': example['metadata']['perturbation_type'],
        'expected_output': example['reference_code'],
        'reason': result['result']
    }

    return result_summary

def main():
    df = pd.read_json("test.jsonl", lines=True)

    # At the beginning, read existing results if any
    output_filename = "dataset_results.jsonl"
    processed_problem_ids = set()
    total_passed = 0
    total_failed = 0
    results_list = []
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            for line in f:
                result = json.loads(line)
                processed_problem_ids.add(result['problem_id'])
                results_list.append(result)
                if result['score'] == 1:
                    total_passed += 1
                else:
                    total_failed += 1

    total_examples = len(df)

    print_environment_info()

    model_id = "../models/llama3_8b_instruct"

    for idx, random_entry in df.iterrows():
        problem_id = random_entry['metadata']['problem_id']
        if problem_id in processed_problem_ids:
            print(f"Skipping already processed problem_id {problem_id}")
            continue
        prompt = random_entry['prompt']

        generated_code = generate_code_from_prompt(model_id, prompt)

        # Save to the same file each time
        save_to_file("dataset_generated_code.py", generated_code)

        processed_code = postprocess(generated_code)
        result = eval_single_example(random_entry, processed_code)

        # Update counts
        if result['score'] == 1:
            total_passed += 1
        else:
            total_failed += 1

        # Collect results
        results_list.append(result)

        # Append the result to the output file
        with open(output_filename, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(f"Processed problem_id {problem_id}")

    # Now calculate and print statistics
    pass_percentage = (total_passed / total_examples) * 100
    fail_percentage = (total_failed / total_examples) * 100

    print("\nStatistics:")
    print(f"Total examples: {total_examples}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Pass percentage: {pass_percentage:.2f}%")
    print(f"Fail percentage: {fail_percentage:.2f}%")

if __name__ == "__main__":
    main()