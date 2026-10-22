"""读字检查：Whisper large-v3 转写中文。"""
import torch

from voicecast.backend.models import DEVICE, model_path
from voicecast.backend.spk import load16k


class ASR:
    def __init__(self, dtype="fp32"):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        from voicecast.backend.tts import DTYPES
        p = model_path("openai/whisper-large-v3")
        self.proc = AutoProcessor.from_pretrained(p)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(p, torch_dtype=DTYPES[dtype]).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, path):
        feats = self.proc(load16k(path), sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(DEVICE, self.model.dtype)
        ids = self.model.generate(feats, language="zh", task="transcribe")
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
