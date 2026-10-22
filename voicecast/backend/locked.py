"""锁定声音 + 逐句演绎描述：官方 generate_voice_clone 不开放 instruct，这里按其流程在底层同时传入
克隆参考（参考音频 + 说话人向量，或仅向量）与 instruct。

- Base 模型：ICL（参考音频 + 文本 + 向量）或仅向量，再加 instruct
- VoiceDesign 模型：没有说话人编码器，但 generate 的说话人槽位可直接接收预先算好的向量
  （由 Base 的 create_voice_clone_prompt(x_vector_only_mode=True) 得到）+ instruct
"""
import numpy as np
import torch


@torch.no_grad()
def generate_locked(tts, texts, prompt_items, instructs, language="Chinese", **gen_kwargs):
    """tts: Qwen3TTSModel（base 或 voice_design）；prompt_items: Base 生成的 VoiceClonePromptItem 列表；
    instructs: 每句的描述（None/"" 表示不加）。返回 (wavs, sr)，与官方接口一致（ICL 时去掉参考部分）。"""
    n = len(texts)
    vcp = tts._prompt_items_to_voice_clone_prompt(prompt_items)
    icl = any(vcp["icl_mode"])
    ref_ids = [tts._tokenize_texts([tts._build_ref_text(it.ref_text)])[0] if it.ref_text else None
               for it in prompt_items] if icl else None
    input_ids = tts._tokenize_texts([tts._build_assistant_text(t) for t in texts])
    instruct_ids = [tts._tokenize_texts([tts._build_instruct_text(s)])[0] if s else None for s in instructs]
    codes, _ = tts.model.generate(input_ids=input_ids, instruct_ids=instruct_ids, ref_ids=ref_ids,
                                  voice_clone_prompt=vcp, languages=[language] * n, non_streaming_mode=False,
                                  **tts._merge_generate_kwargs(**gen_kwargs))
    ref_codes = vcp.get("ref_code") or [None] * n
    full = [torch.cat([r.to(c.device), c], 0) if r is not None else c for r, c in zip(ref_codes, codes)]
    wavs, sr = tts.model.speech_tokenizer.decode([{"audio_codes": c} for c in full])
    out = []
    for r, c, w in zip(ref_codes, full, wavs):
        cut = int(r.shape[0] / max(c.shape[0], 1) * w.shape[0]) if r is not None else 0
        out.append(np.asarray(w[cut:], dtype=np.float32))
    return out, sr
