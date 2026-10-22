"""说话人向量：CAM++（funasr），16 kHz 输入、L2 归一化输出。两个声音的相似度 = 向量点积。"""
import librosa
import numpy as np
import torch
import torch.nn.functional as F

from voicecast.backend.models import DEVICE, model_path


def load16k(path):
    return librosa.load(path, sr=16000)[0]


class CAMPPlus:
    def __init__(self):
        from funasr import AutoModel
        self.model = AutoModel(model=model_path("funasr/campplus"), device=DEVICE, disable_update=True,
                               disable_pbar=True, disable_log=True)

    @torch.no_grad()
    def __call__(self, y):
        return self.model.generate(input=y)[0]["spk_embedding"][0]


def embed(model, paths):
    """→ (len(paths), D) 的 L2 归一化向量。"""
    return np.stack([F.normalize(model(load16k(p)).float(), dim=-1).cpu().numpy() for p in paths])
