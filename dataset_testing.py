import os
import random
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


def append_to_json(filename, data):
    try:
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r') as file:
                try:
                    json_data = json.load(file)
                except json.JSONDecodeError:
                    print(f"File {filename} is corrupted, initializing with an empty list.")
                    json_data = []
        else:
            json_data = []

        json_data.append(data)

        with open(filename, 'w') as file:
            json.dump(json_data, file, indent=4)
    except Exception as e:
        print(f"Error writing to {filename}: {e}")


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
        'reason':result['result']
    }

    return result_summary


def main():
    df = pd.read_json("test.jsonl", lines=True)

    random_entry = df.sample(n=1).iloc[0]
    prompt = random_entry['prompt']

    print_environment_info()

    model_id = "../models/llama3_8b_instruct"

    generated_code = generate_code_from_prompt(model_id, prompt)

    save_to_file("dataset_generated_code.py", generated_code)

    processed_code = postprocess(generated_code)
    result = eval_single_example(random_entry, processed_code)

    print("\nGenerated Code:")
    print(processed_code)
    print("\nEvaluation Result:")
    print(f"Score: {result['score']}, because {result['reason']}")
    print(f"Expected Output: {result['expected_output']}")
    print(f"Library: {result['library']}")
    print(f"Perturbation Type: {result['perturbation_type']}")

    output_filename = "dataset_results.json"
    append_to_json(output_filename, result)
    print(f"Result appended to {output_filename}")


if __name__ == "__main__":
    main()

#chain of thought§