# Model Download Instructions

The models used in this project are too large to be stored in this repository. To replicate the results, please download the specific GGUF model files listed below and place them into the `models/` directory at the root of this project.

**Important:** To ensure reproducibility, it is crucial to use the exact quantization specified (`Q4_K_M`). Different quantizations may lead to variations in the results.

## Download Links

| Model Name | Filename Used in Code | Direct GGUF Download Page |
| :--- | :--- | :--- |
| **Hermes-3-Llama-3.2-3B** | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | [Download Link](https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B.Q4_K_M.gguf) |
| **Mistral-7B-Instruct-v0.1** | `mistral-7b-instruct-v0.1.Q4_K_M.gguf` | [Download Link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf) |
| **OpenChat 3.5** | `openchat-3.5-0106.Q4_K_M.gguf` | [Download Link](https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/blob/main/openchat-3.5-0106.Q4_K_M.gguf) |
| **StableLM Zephyr 3B** | `stablelm-zephyr-3b.Q4_K_M.gguf` | [Download Link](https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/blob/main/stablelm-zephyr-3b.Q4_K_M.gguf) |
| **TinyLLaMA-1.1B-Chat-v1.0** | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | [Download Link](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf) |

### Instructions

1.  Click on the "Download Link" for each model.
2.  Navigate to the "**Files and versions**" tab on the Hugging Face page.
3.  Find the specific file ending in **`Q4_K_M.gguf`** and download it.
4.  Place the downloaded file into the `models/` directory of this project.
5.  Ensure the filename exactly matches the one listed in the "Filename Used in Code" column. You may need to rename the file after downloading.