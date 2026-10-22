"""引擎：类型库与提示词、TTS 模型（设计 / 锁定 / 生成）、打分（音色、情绪、清晰度、读字）。"""
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from voicecast.backend.emotion import Scorer, pick
from voicecast.backend.locked import generate_locked
from voicecast.backend.measure import clip_attrs, norm_text
from voicecast.backend.models import BATCH, DTYPE
from voicecast.backend.tts import load_tts, set_seed
from voicecast.library import LIBRARY

REF_LINE = "你好，很高兴认识你。我平时喜欢看书，也喜欢出去走走。"
TOKENS_PER_S = 12.5


def max_tokens(text):
    """生成上限：与时长异常检查一致（0.5 s/字 + 3 s），防止失控生成。"""
    return int(TOKENS_PER_S * (0.5 * len(norm_text(text)) + 3))


def render(vtype, card):
    """参数卡 → 提示词（固定模板）。card: {pitch, roughness, breathiness, resonance, weight, nasality}"""
    t, a = LIBRARY["types"][vtype], LIBRARY["attributes"]
    return (f"{t['identity_base']}, {t['pitch'][card['pitch']]} pitch, {a['roughness'][card['roughness']]}, "
            f"{a['breathiness'][card['breathiness']]} timbre with {a['resonance'][card['resonance']]}, "
            f"{a['weight'][card['weight']]}{a['nasality'][card['nasality']]}; {t['delivery']}.")


def validate_card(vtype, card):
    """参数卡校验：类型存在、每维取值合法。返回问题列表。"""
    if vtype not in LIBRARY["types"]:
        return [f"未知类型 {vtype}"]
    dims = ["pitch", *LIBRARY["attributes"]]
    return [f"{d}={card.get(d)!r} 不合法" for d in dims if card.get(d) not in ("lo", "hi")]


def random_cards(n, seed):
    """n 张互不相同的随机参数卡（6 维 × 2 档）。"""
    rng = random.Random(seed)
    dims = ["pitch", *LIBRARY["attributes"]]
    seen, cards = set(), []
    while len(cards) < n:
        c = tuple(rng.choice(["lo", "hi"]) for _ in dims)
        if c not in seen or len(seen) >= 2 ** len(dims):
            seen.add(c)
            cards.append(dict(zip(dims, c)))
    return cards


class Engine:
    """按需加载模型：design（选角时）、base（锁定与配音）、scorer（测量）。"""

    def __init__(self, dtype=None, batch=None):
        self.dtype = dtype or DTYPE
        self.batch = batch or BATCH
        self._design = self._base = self._scorer = self._asr = None

    @property
    def design(self):
        if self._design is None:
            self._design = load_tts("design", self.dtype)
        return self._design

    @property
    def base(self):
        if self._base is None:
            self._base = load_tts("base", self.dtype)
        return self._base

    @property
    def scorer(self):
        if self._scorer is None:
            self._scorer = Scorer()
        return self._scorer

    def release_design(self):
        import torch
        self._design = None
        torch.cuda.empty_cache()

    def design_refs(self, prompts, seeds, text=REF_LINE, batch=None, on_batch=None):
        """每个提示词 + 种子生成一句参考音。返回 (wavs, sr)。同一种子的提示词一起成批生成
        （同一批内即使提示词相同，生成的也是不同的声音）。text 可为每项各自的台词列表。
        on_batch(已完成数, 总数)：每批完成后回调（进度条用）。"""
        batch = batch or self.batch
        texts = text if isinstance(text, list) else [text] * len(prompts)
        out = [None] * len(prompts)
        sr = None
        done = 0
        for s in sorted(set(seeds)):
            idx = [i for i, x in enumerate(seeds) if x == s]
            set_seed(s)
            for j in range(0, len(idx), batch):
                b = idx[j:j + batch]
                wavs, sr = self.design.generate_voice_design(text=[texts[i] for i in b], instruct=[prompts[i] for i in b],
                                                             language=["Chinese"] * len(b),
                                                             max_new_tokens=max(max_tokens(texts[i]) for i in b))
                for i, w in zip(b, wavs):
                    out[i] = np.asarray(w, dtype=np.float32)
                done += len(b)
                if on_batch:
                    on_batch(done, len(prompts))
        return out, sr

    def lock(self, ref_path, ref_text=REF_LINE):
        """锁定：ICL（参考音频 + 文本 + 向量）与仅向量两种克隆参考。"""
        icl = self.base.create_voice_clone_prompt(ref_audio=str(ref_path), ref_text=ref_text)[0]
        xv = self.base.create_voice_clone_prompt(ref_audio=str(ref_path), x_vector_only_mode=True)[0]
        return icl, xv

    def say(self, jobs, seed, batch=None):
        """jobs: [{text, lock: (icl, xv), emotion}] → wavs。平静用 ICL 克隆；带情绪用 M3（仅向量 + 情绪描述）。"""
        batch = batch or self.batch
        set_seed(seed)
        out, sr = [None] * len(jobs), None
        for kind in ("icl", "m3"):
            idx = [i for i, j in enumerate(jobs) if (kind == "icl") == (not LIBRARY["emotions"].get(j["emotion"] or "neutral"))]
            for k in range(0, len(idx), batch):
                b = idx[k:k + batch]
                items = [jobs[i]["lock"][0 if kind == "icl" else 1] for i in b]
                ins = [LIBRARY["emotions"].get(jobs[i]["emotion"] or "neutral") or None for i in b]
                cap = max(max_tokens(jobs[i]["text"]) for i in b)
                wavs, sr = generate_locked(self.base, [jobs[i]["text"] for i in b], items, ins, "Chinese",
                                           max_new_tokens=cap)
                for i, w in zip(b, wavs):
                    out[i] = w
        return out, sr

    def asr(self, path):
        if self._asr is None:
            from voicecast.backend.asr import ASR
            self._asr = ASR(self.dtype)
        return self._asr(path)

    def cer(self, path, text):
        import jiwer
        hyp = norm_text(self.asr(path))
        return jiwer.cer(norm_text(text), hyp) if hyp else 1.0


def hard_check(path, text):
    """硬检查：时长异常（失控生成）、静音过多、爆音。返回问题列表。"""
    a = clip_attrs(path)
    probs = []
    if a["dur_s"] > 0.5 * len(norm_text(text)) + 3:
        probs.append("时长异常")
    if a["silence_frac"] > 0.5:
        probs.append("静音过多")
    if a["clip_frac"] > 0.001:
        probs.append("爆音")
    return probs, a


__all__ = ["Engine", "LIBRARY", "REF_LINE", "render", "validate_card", "random_cards", "hard_check", "pick",
           "max_tokens"]
