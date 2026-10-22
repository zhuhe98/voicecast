"""Qwen3-TTS 加载（VoiceDesign 设计声音、Base 克隆与锁定生成）。Turing 等不支持 bf16 的显卡用 fp32。"""
import random

import numpy as np
import torch

from voicecast.backend.models import DEVICE, model_path

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
REPOS = {"design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}


def load_tts(kind, dtype="fp32"):
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(model_path(REPOS[kind]), device_map=DEVICE,
                                         dtype=DTYPES[dtype], attn_implementation="sdpa")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
