"""Compute the training loss from the answer's logits only, so Qwen3 / Qwen3-VL QLoRA fits a 4GB card.

The model normally scores every position against its 151,936-word vocabulary, image tokens and prompt included, then
upcasts that to float32 for the loss: about 1GB for a 1,000-token page. The loss only uses positions whose label isn't
-100 (the answer), so this keeps logits from the first answer position on. The loss value is the same as the default
(mean cross-entropy over answer tokens); only memory changes.
"""

import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_vl import modeling_qwen3_vl

IGNORE = -100
CHUNK = 128  # answer positions per loss chunk
CLASSES = [modeling_qwen3_vl.Qwen3VLForConditionalGeneration, modeling_qwen3.Qwen3ForCausalLM]
CALLS = 0  # how many training forwards went through the answer-only path


def _wrap(original):
    def forward(self, *args, labels=None, logits_to_keep=0, **kwargs):
        if labels is None or not self.training:
            return original(self, *args, labels=labels, logits_to_keep=logits_to_keep, **kwargs)

        global CALLS
        CALLS += 1
        length = labels.shape[1]
        first = int(labels.ne(IGNORE).any(0).nonzero()[0])  # first answer position across the batch
        keep = min(length, length - first + 1)  # one more, because position p predicts token p + 1
        outputs = original(self, *args, labels=None, logits_to_keep=keep, **kwargs)

        logits = outputs.logits[:, :-1]
        targets = labels[:, length - keep + 1:]
        # Float32 log-softmax over 151,936 words for a whole answer is another ~200MB twice over, enough to tip a
        # 4GB card. Do it CHUNK positions at a time, recomputed in backward instead of stored. Same mean loss.
        total = sum(checkpoint(_chunk_loss, logits[:, i:i + CHUNK], targets[:, i:i + CHUNK], use_reentrant=False)
                    for i in range(0, logits.shape[1], CHUNK))
        outputs.loss = total / targets.ne(IGNORE).sum().clamp(min=1)
        return outputs

    forward.answer_only = True
    return forward


def _chunk_loss(logits, targets):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE, reduction="sum")


def install() -> None:
    for cls in CLASSES:
        if not getattr(cls.forward, "answer_only", False):
            cls.forward = _wrap(cls.forward)


def installed() -> str:
    """For logs: whether the patch is in place and how many training steps actually used it."""
    in_place = all(getattr(cls.forward, "answer_only", False) for cls in CLASSES)
    return f"{in_place} ({CALLS} calls)"
