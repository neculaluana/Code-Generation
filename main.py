import os
import subprocess
import time

import transformers
import torch


def is_virtual_environment():
    return os.getenv('VIRTUAL_ENV') is not None


print(f"Torch version: {torch.__version__}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
if is_virtual_environment():
    print("You are inside a virtual environment.")
else:
    print("You are not inside a virtual environment.")
with open("context.txt", 'w') as file:
    print("Context resert.")


model_pipeline = None

def get_pipeline(model_id):
    global model_pipeline
    if model_pipeline is None:
        model_pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
    return model_pipeline


def execute_pipeline(model_id, messages, mode):
    """ Create an instance of HuggingFacePipeline with chat configuration """
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


def save_to_file(filename, content):
    """Save generated content to a file"""
    content=content.replace('`', '')
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Saved {filename}")


def run_python_script(filename):
    """Run a Python script and capture its output."""
    try:
        result = subprocess.run(['python', filename], capture_output=True, text=True, check=True)
        if result.returncode == 0:
            print("All tests passed.")
            return result.stdout, result.stderr, False
        else:
            print("There were some errors in the tests.")
            print(result.stdout)
            return result.stdout, result.stderr, True

    except subprocess.CalledProcessError as e:
        print("Error running script:", e)
        print(e.stdout)
        print(e.stderr)
        return e.stdout, e.stderr, True

def append_to_context(content, section_title):
    """Append content to the context file under a specific section"""
    with open("context.txt", 'a') as file:
        file.write(f"\n## {section_title}\n")
        file.write(content)
    print(f"Appended {section_title} to context.txt")



def fix_code(model_id, code, tests, errors):
    """Use the model to fix the code based on the errors."""
    fix_messages = [
    {
        "role": "system",
        "content": "You are a senior Python developer that receives error messages and the code, and you need to fix the errors in the code. You are allowed to write only the code with no explanations."
    },
    {
        "role": "user",
        "content": f"""
        Here is the code that needs to be fixed:
        {code}

        Here are the tests that identify the errors:
        {tests}

        Here are the error messages encountered during testing:
        {errors}

        Instructions:
        1. Carefully analyze the provided code and error messages.
        2. Apply necessary fixes to ensure the code passes all the provided tests.
        3. Ensure the code follows best practices and handles edge cases appropriately.
        4. Validate all inputs and handle possible errors gracefully to prevent crashes.
        5. Make sure the code is efficient and optimized for performance.
        6. Provide only the fixed code without explanations or comments.
        7. Ensure the fixed code is runnable and correct.
        """
    }
]

    fixed_code = execute_pipeline(model_id, fix_messages, "fix")
    return fixed_code

def generate_new_code(model_id, context, additional_instructions=""):
    """Use the model to generate new code based on the context and additional instructions."""
    generate_messages = [
        {
            "role": "system",
            "content": "You are a senior Python developer that receives context and additional instructions to generate new code. You are allowed to write only the code with no explanations."
        },
        {
            "role": "user",
            "content": f"""
            Here is the current context:
            {context}

            Additional instructions:
            {additional_instructions}

            Instructions:
            1. Carefully analyze the provided context and additional instructions.
            2. Generate new code based on the instructions, integrating it with the existing context.
            3. Ensure the code follows best practices and handles edge cases appropriately.
            4. Validate all inputs and handle possible errors gracefully to prevent crashes.
            5. Make sure the code is efficient and optimized for performance.
            6. Provide only the new code without explanations or comments.
            7. Ensure the new code is runnable and correct.
            """
        }
    ]

    new_code = execute_pipeline(model_id, generate_messages, "generate")
    return new_code

def generate_new_tests(model_id, code, context, additional_instructions=""):
    """Use the model to generate new tests based on the code, context, and additional instructions."""
    generate_messages = [
        {
            "role": "system",
            "content": "You are a senior Python developer that receives code, context, and additional instructions to generate new unit tests. You are allowed to write only the test code with no explanations."
        },
        {
            "role": "user",
            "content": f"""
            Here is the code that needs to be tested:
            {code}

            Here is the current context:
            {context}

            Additional instructions:
            {additional_instructions}

           Instructions:
            0. Begin your response with the necessary imports and setup, including the import for the generated_code module.
            1. Write comprehensive unit tests to cover all functions and methods.
            2. Include tests for the most common possible errors.
            3. Ensure the tests handle scenarios that may require specific conditions.
            4. Ensure all the tests are logically and mathematically correct.
            5. Use appropriate test frameworks like unittest or pytest.
            6. Make sure the tests are runnable and correct.
            7. Provide only the test code without explanations, comments, or including the input code.
            8. Keep the total token usage within 8000 tokens, including the provided code.
            """
        }
    ]

    new_tests = execute_pipeline(model_id, generate_messages, "tests")
    return new_tests



model_id = "../models/llama3_8b_instruct"
prompt = "Write a function to check if a given string is a palindrome."

code_messages = [
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

code_output = execute_pipeline(model_id, code_messages, "code")
append_to_context(code_output, "Original Code")
save_to_file("generated_code.py", code_output)
print(code_output)

test_messages = [
    {
        "role": "system",
        "content": "You are an expert Python code tester. Your task is to write the necessary tests for the given code in the module generated_code. Provide only the test code. Avoid any explanations, comments, or including the input code."
    },
    {
        "role": "user",
        "content": f"""
        Here is the code snippet for which you need to write tests:
        {code_output}

        Instructions:
        0. Begin your response with the necessary imports and setup, including the import for the generated_code module.
        1. Write comprehensive unit tests to cover all functions and methods.
        2. Include tests for the most common possible errors.
        3. Ensure the tests handle scenarios that may require specific conditions.
        4. Ensure all the tests are logically and mathematically correct.
        5. Use appropriate test frameworks like unittest or pytest.
        6. Make sure the tests are runnable and correct.
        7. Provide only the test code without explanations, comments, or including the input code.
        8. Keep the total token usage within 8000 tokens, including the provided code.
        """
    }
]
test_output = execute_pipeline(model_id, test_messages, "test")
save_to_file("testing.py", test_output)
append_to_context(test_output, "Tests")
print(test_output)


stdout, stderr, test_failed = run_python_script("testing.py")
loop_index = 1


while test_failed and loop_index <= 5:
    errors = f"{stdout}\n{stderr}"
    with open("generated_code.py", "r") as file:
        code_content = file.read()
    with open("testing.py", "r") as file:
        test_content = file.read()
    if test_failed:
        fixed_code = fix_code(model_id, code_content, test_content, errors)
        save_to_file("generated_code.py", fixed_code)
        append_to_context(fixed_code, f"Fix Iteration {loop_index}")
    print(fixed_code)
    stdout, stderr, test_failed = run_python_script("testing.py")
    append_to_context(stdout + stderr, "Test Output")
    loop_index = loop_index + 1

append_to_context(stdout + stderr, "Final Test Output")

additional_instructions = "Modify the code so it also works for integers"

if additional_instructions.strip():
    with open("context.txt", "r") as file:
        current_context = file.read()
    new_code = generate_new_code(model_id, current_context, additional_instructions)
    save_to_file("generated_code.py", new_code)
    append_to_context(new_code, "Additional Instructions Code")
    print("New Code with Additional Instructions:")
    print(new_code)
    new_tests = generate_new_tests(model_id, new_code,current_context, additional_instructions)
    save_to_file("testing.py", new_tests)
    print("_________________________________________________")
    print("New Tests with Additional Instructions:")
    print(new_tests)
else:
    print("No additional instructions provided.")

stdout, stderr, test_failed = run_python_script("testing.py")
loop_index = 1


while test_failed and loop_index <= 5:
    errors = f"{stdout}\n{stderr}"
    with open("generated_code.py", "r") as file:
        code_content = file.read()
    with open("testing.py", "r") as file:
        test_content = file.read()
    if test_failed:
        fixed_code = fix_code(model_id, code_content, test_content, errors)
        save_to_file("generated_code.py", fixed_code)
        append_to_context(fixed_code, f"Fix Iteration {loop_index}")
    print(fixed_code)
    stdout, stderr, test_failed = run_python_script("testing.py")
    append_to_context(stdout + stderr, "Test Output")
    loop_index = loop_index + 1