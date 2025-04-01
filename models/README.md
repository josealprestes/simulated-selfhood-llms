The models used in this project are too large to be stored in this repository. Please download them manually from the links below and place the files into this `models/` directory.

All models are in `.gguf` format, compatible with `llama-cpp-python`.

## Download Links

- **Hermes-3-LLaMA-3.2B**  
  [Download](https://huggingface.co/NousResearch/Hermes-3-llama-2-3b)  

- **Mistral-7B-Instruct-v0.1**  
  [Download](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

- **OpenChat 3.5**  
  [Download](https://huggingface.co/openchat/openchat-3.5)

- **StableLM Zephyr 3B**  
  [Download](https://huggingface.co/stabilityai/stablelm-zephyr-3b)

- **TinyLLaMA-1.1B-Chat**  
  [Download](https://huggingface.co/cognitivecomputations/TinyLlama-1.1B-Chat-v1.0)

After downloading, make sure the filenames match those referenced in the code (e.g., `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf`).

## Note

You may need to convert the models to GGUF format using the `transformers` or `llama.cpp` toolchain if not already provided.