# Changes to third-party repos

The repos themselves are not included. Clone them at these commits:

- LlamaFactory: https://github.com/hiyouga/LlamaFactory (commit 9997aa07)
- llama.cpp: https://github.com/ggml-org/llama.cpp (commit fac3a89), no changes
- ThaiOCRBench: https://github.com/scb-10x/ThaiOCRBench (commit c6f3959)

## LlamaFactory
- `modified_files.patch`: changes to `data/dataset_info.json` and `data/identity.json` (apply with `git apply`)
- `data/`: the modified files and the new training datasets (`thai_doc_*.json`)
- `models/`: config/tokenizer/Modelfile of the trained models only (weights `.safetensors`/`.gguf` are not included because they exceed GitHub's limit)

## ThaiOCRBench
- `ollama_eval/`, `ollama_pred/`, `ollama_res/`: scripts and evaluation results with Ollama
