r"""Convert a Qwen3-VL model folder (or Hugging Face id) to GGUF and register it in Ollama, vision included.

    ..\LlamaFactory\.venv\Scripts\python.exe to_ollama.py Qwen/Qwen3-VL-2B-Instruct qwen3vl-base-q8
    ..\LlamaFactory\.venv\Scripts\python.exe to_ollama.py ..\LlamaFactory\models\qwen3vl-2b-thaidoc qwen3vl-thaidoc-q8

Makes two files with llama.cpp's converter, the language model (q8_0) and the vision projector (mmproj, f16), and an
Ollama Modelfile with both, the way Ollama stores the Typhoon import. Convert the base and the trained model the same
way, so a comparison only differs in weights. Needs the LlamaFactory venv (torch and gguf-py are installed there).
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CONVERTER = HERE.parent / "llama.cpp" / "convert_hf_to_gguf.py"
OUTPUT = HERE.parent / "LlamaFactory" / "models" / "gguf"

MODELFILE = '''FROM ./model.gguf
FROM ./mmproj.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
'''

# Text-only Qwen3 (hybrid thinking): an empty <think></think> after each user turn switches thinking off, the same
# as LlamaFactory's qwen3 template with enable_thinking false. Used for text-to-JSON models.
TEXT_MODELFILE = '''FROM ./model.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
<|im_start|>assistant
<think>

</think>

{{ else if eq .Role "assistant" }}{{ .Content }}<|im_end|>
{{ end }}{{ end }}"""
PARAMETER stop "<|im_end|>"
'''


def source_folder(source: str) -> Path:
    path = Path(source)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download  # a Hugging Face id that is already downloaded
    return Path(snapshot_download(source, local_files_only=True))


def run(*command) -> None:
    print(">", " ".join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="model folder, or a downloaded Hugging Face id")
    parser.add_argument("name", help="Ollama model name")
    parser.add_argument("--outtype", default="q8_0", help="language model precision (q8_0, f16, bf16)")
    parser.add_argument("--text", action="store_true", help="text-only Qwen3: no vision projector, thinking off")
    args = parser.parse_args()

    source = source_folder(args.source)
    folder = OUTPUT / args.name
    folder.mkdir(parents=True, exist_ok=True)
    run(sys.executable, CONVERTER, source, "--outtype", args.outtype, "--outfile", folder / "model.gguf")
    if not args.text:
        run(sys.executable, CONVERTER, source, "--mmproj", "--outtype", "f16", "--outfile", folder / "mmproj.gguf")
        if not (folder / "mmproj.gguf").exists():  # some versions add an mmproj- prefix to the name given
            for candidate in folder.glob("mmproj-*.gguf"):
                candidate.replace(folder / "mmproj.gguf")
    (folder / "Modelfile").write_text(TEXT_MODELFILE if args.text else MODELFILE, encoding="utf-8")
    subprocess.run(["ollama", "create", args.name, "-f", "Modelfile"], cwd=folder, check=True)
    print(f"Ollama model '{args.name}' created from {source}")


if __name__ == "__main__":
    main()
