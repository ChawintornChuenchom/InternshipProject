r"""Run a LlamaFactory training config with the answer-only loss installed. From the LlamaFactory folder:

    .venv\Scripts\python.exe ..\thai-doc-ocr\train.py ..\thai-doc-ocr\configs\qwen3vl_2b_qlora_train.yaml

Same as `llamafactory-cli train <config>`, plus answer_only_loss: without it, Qwen3-VL-2B QLoRA needs 3.3-4.2GB and
crashes a 4GB card after a few steps; with it, a full-resolution page peaks around 2.5GB. Prints GPU memory after
each optimizer step, so running past the card (Windows silently spills into RAM and slows down) shows up early.
"""

import os
import sys
from pathlib import Path

import torch
from transformers import TrainerCallback

sys.path.insert(0, str(Path(__file__).parent))

import answer_only_loss
from llamafactory.train.tuner import run_exp

# Windows never reports out-of-memory: past the card it quietly lends system RAM (slow, and the earlier runs died with
# CUBLAS errors). So PyTorch never frees its cache and keeps growing: 2.8GB in use, 4.4-5.4GB reserved. A cap makes it
# hit a limit, free cached blocks and retry, as it would on Linux. 0.75 of 4GB = 3,072MB; a Qwen3-VL step uses ~2,600MB.
GPU_FRACTION = float(os.environ.get("TRAIN_GPU_FRACTION", "0.75"))


class MemoryReport(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        # Only from here: loading Qwen3-1.7B briefly builds a 1.16GB float32 output layer before tying it to the
        # embeddings, which a cap set at start-up turned into an out-of-memory error.
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)

    def on_step_end(self, args, state, control, **kwargs):
        mib = 2**20
        print(f"[memory] step {state.global_step}: peak allocated {torch.cuda.max_memory_allocated() / mib:.0f} MiB, "
              f"reserved {torch.cuda.memory_reserved() / mib:.0f} MiB, patched={answer_only_loss.installed()}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()  # hand back what the step left cached, so the next one starts small


if __name__ == "__main__":
    answer_only_loss.install()
    run_exp(callbacks=[MemoryReport()])  # reads the YAML path (and any key=value overrides) from the command line
