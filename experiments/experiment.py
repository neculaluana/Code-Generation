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


def fix_code(model_id, code, tests, errors):
    """Use the model to fix the code based on the errors."""
    fix_messages = [
            {"role": "system", "content": "You are a senior python developer that receives error messages and the code, and you need to fix the errors in the code. You are allowed to write only code with no explainations."},
            {"role": "user", "content": f"Here is the code:\n{code}\nHere are the tests:\n{tests}\nHere are the errors:\n{errors}"}
        ]

    fixed_code = execute_pipeline(model_id, fix_messages, "fix")
    return fixed_code



model_id = "../../models/llama3_8b_instruct"
prompt = "Write a function that checks if a given number is a prime number."

code_messages = [
    {"role": "system",
     "content": "You are an senior python developer that receives instructions and write code based on them. Write only code with no explanations."},
    {"role": "user", "content": prompt}
]

code_output = execute_pipeline(model_id, code_messages, "code")
save_to_file("../generated_code.py", code_output)
print(code_output)

test_messages = [
    {"role": "system",
     "content": "You are an expert python code tester. Write the necessary tests for the following code and run them. Write only code with no explanations.The code is located in the module generated_code."},
    {"role": "user", "content": f"Assuming the following code snippet, write the necessary tests to account for possible errors and crashes cases, and cases that would need a lot of processing power: {code_output}"}
]

test_output = execute_pipeline(model_id, test_messages, "test")
save_to_file("../testing.py", test_output)
print(test_output)

stdout, stderr, test_failed = run_python_script("../testing.py")
loop_index = 1


while test_failed and loop_index <= 5:
    errors = f"{stdout}\n ~~~~~~~~~~~~~~~~~~~~\n{stderr}"
    save_to_file("errors.txt", errors)
    with open("../generated_code.py", "r") as file:
        code_content = file.read()
    with open("../testing.py", "r") as file:
        test_content = file.read()
    if test_failed:
        fixed_code = fix_code(model_id, code_content, test_content, errors)
        save_to_file("../generated_code.py", fixed_code)
    print(fixed_code)
    stdout, stderr, test_failed = run_python_script("../testing.py")
    loop_index = loop_index + 1
