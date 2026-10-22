"""情绪与质量打分：emotion2vec 情绪概率、CAM++ 音色相似度、CPPS 清晰度变化；逐句候选挑选规则。"""
import numpy as np

from voicecast.backend.measure import clip_attrs
from voicecast.backend.models import DEVICE, model_path
from voicecast.backend.spk import CAMPPlus, embed

EMOTIONS = ["neutral", "happy", "angry", "sad", "surprised"]


class Scorer:
    def __init__(self):
        from funasr import AutoModel
        self.emo = AutoModel(model=model_path("emotion2vec/emotion2vec_plus_large"), device=DEVICE,
                             disable_update=True, disable_pbar=True, disable_log=True)
        self.cam = CAMPPlus()

    def emotion(self, path):
        r = self.emo.generate(str(path), granularity="utterance", extract_embedding=False)[0]
        return {l.split("/")[-1]: float(s) for l, s in zip(r["labels"], r["scores"])}

    def baseline(self, paths):
        """该声音中性台词 → (CAM++ 质心, 平均 CPPS)。"""
        E = embed(self.cam, paths)
        c = E.mean(0)
        return c / np.linalg.norm(c), float(np.mean([clip_attrs(p)["cpps"] for p in paths]))

    def score(self, path, target, base):
        """target 情绪；base = baseline(...)。返回 dict(p_target, pred, timbre_sim, dcpps)。"""
        p = self.emotion(path)
        e = embed(self.cam, [path])[0]
        return {"p_target": p.get(target, 0.0), "pred": max(p, key=p.get),
                "timbre_sim": float(e @ base[0]), "dcpps": clip_attrs(path)["cpps"] - base[1]}


def pick(cands, min_timbre, min_dcpps):
    """候选挑选：先满足 音色 ≥ min_timbre 且 清晰度变化 ≥ min_dcpps，再取目标情绪概率最高；
    都不满足时取 音色 + 情绪概率 之和最高。cands: [dict(p_target, timbre_sim, dcpps, ...)]，返回下标。"""
    ok = [i for i, c in enumerate(cands) if c["timbre_sim"] >= min_timbre and c["dcpps"] >= min_dcpps]
    if ok:
        return max(ok, key=lambda i: cands[i]["p_target"])
    return max(range(len(cands)), key=lambda i: cands[i]["timbre_sim"] + cands[i]["p_target"])
