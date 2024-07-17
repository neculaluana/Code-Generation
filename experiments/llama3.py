import time

import transformers
import torch
import os
from transformers import pipeline, AutoModel, AutoTokenizer

def is_virtual_environment():
    return os.getenv('VIRTUAL_ENV') is not None

# Example usage:
if is_virtual_environment():
    print("You are inside a virtual environment.")
else:
    print("You are not inside a virtual environment.")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#llama3_8b_instruct
#llama3_model = AutoModel.from_pretrained('llama3_8b_instruct')
#llama3_tokenizer = AutoTokenizer.from_pretrained('llama3_8b_instruct')

def read_token(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Token file not found.")
        return None

pipeline = transformers.pipeline(
    "text-generation",
    model="../models/llama3_8b_instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)
#pipeline.save_pretrained('llama3_8b_instruct')
prompt = "write a function in python to check if i am running the code inside the virtual environment? i'm working in pycharm"
messages = [
    {"role": "system", "content": "You are an senior python developer that receives instructions and write code based on them. Write only code with no explainations."},
    {"role": "user", "content": prompt},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
start_time = time.time()
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
end_time = time.time()
print(outputs[0]["generated_text"][len(prompt):])

print(f"Time taken: {end_time-start_time}")

