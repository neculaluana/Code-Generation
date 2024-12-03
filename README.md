# Code-Generation

This project automates Python code generation, testing, and debugging using the Hugging Face Transformers library and PyTorch. It leverages state-of-the-art text-generation models to create functional code, generate unit tests, and iteratively debug and fix errors.

## Technologies Used

- **Hugging Face Transformers**: For leveraging pre-trained models for text generation.
- **PyTorch**: Used for efficient computation and model deployment, including GPU acceleration.
- **Python**: Core programming language for scripting and orchestration.

## Key Features and Logic

1. **Environment Setup**:
   - Checks for a virtual environment and verifies CUDA compatibility.
   - Prints versions of key components for reproducibility.

2. **Pipeline Management**:
   - Initializes a Hugging Face pipeline for text generation with specified configurations (e.g., model dtype and device mapping).

3. **Code Generation**:
   - Takes a natural language prompt to generate Python code.
   - Saves the generated code to a file and appends it to the context log.

4. **Test Generation**:
   - Creates unit tests for the generated code using a model-based approach.

5. **Debugging and Fixing**:
   - Runs the tests and captures errors.
   - Uses the model to fix the code iteratively based on the test output and error logs.
   - Repeats the cycle up to 5 times to ensure correctness.

6. **Additional Functionality**:
   - Supports generating new code or tests based on additional instructions.
   - Appends all modifications to a context file for traceability.
