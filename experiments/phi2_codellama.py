import sys
import transformers
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TRANSFORMERS_CACHE
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

class ModelHandler:
    def __init__(self, model_name, save_path):
        self.model_name = model_name
        self.save_path = save_path
        self.model = None
        self.tokenizer = None

    def download_and_save_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

    def load_model_from_local(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.save_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_path, trust_remote_code=True)

    def get_pipeline(self):
        return transformers.pipeline(model=self.save_path, task="text-generation", max_new_tokens=300)



class SequentialChainHandler:
    def __init__(self, phi2_pipeline, codellama_pipeline):
        self.phi2_pipeline = phi2_pipeline
        self.codellama_pipeline = codellama_pipeline

    def create_chains(self):
        phi2_template = PromptTemplate(
            input_variables=["user_input"],
            template="""
                Instruct:Please provide a comprehensive and clear explanation of the steps required to accomplish the following task, emphasizing efficient strategies and key considerations. Avoid generating any code. Focus solely on detailing the approach and methodologies that would be applied to execute this task effectively.After providing the steps for execution, conclude the output with the symbol ~ to indicate the end of the sequence.\nTask:{user_input}\nSteps for Execution:
                """
        )

        llama_template = PromptTemplate(
            input_variables=["phi2"],
            template="<s>[INST] {phi2} [/INST]"
        )

        phi2_hf = HuggingFacePipeline(pipeline=self.phi2_pipeline)
        codellama_hf = HuggingFacePipeline(pipeline=self.codellama_pipeline)

        phi_chain = LLMChain(llm=phi2_hf, prompt=phi2_template, output_key="phi2")
        llama_chain = LLMChain(llm=codellama_hf, prompt=llama_template, output_key="codellama")

        self.chain = SequentialChain(chains=[phi_chain, llama_chain], input_variables=["user_input"], output_variables=["phi2", "codellama"])

    def run_chain(self, user_input):
        return self.chain.invoke({"user_input": user_input})


def print_system_info():
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Transformers Cache: {TRANSFORMERS_CACHE}")

def main():
    print_system_info()

    # Define model paths
    codellama_model_id = "codellama/CodeLlama-7b-Instruct-hf"
    codellama_save_path = '../models/codellamaInstruct'
    phi2_model_id = "microsoft/phi-2"
    phi2_save_path = '../phi2Model'

    # Instantiate model handlers
    codellama_handler = ModelHandler(codellama_model_id, codellama_save_path)
    phi2_handler = ModelHandler(phi2_model_id, phi2_save_path)

    # Download and save models locally
    #codellama_handler.download_and_save_model()
    #phi2_handler.download_and_save_model()

    # Load models from local directory
    # codellama_handler.load_model_from_local()
    # phi2_handler.load_model_from_local()

    # Create pipelines
    codellama_pipeline = codellama_handler.get_pipeline()
    phi2_pipeline = phi2_handler.get_pipeline()

    # Create and run sequential chain
    chain_handler = SequentialChainHandler(phi2_pipeline, codellama_pipeline)
    chain_handler.create_chains()

    user_input = "write a function that computes Fibonacci in Python"
    start_time = time.time()
    result = chain_handler.run_chain(user_input)
    end_time = time.time()

    # Print results
    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
