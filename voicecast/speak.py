"""逐句配音：每句 n 个候选（平静 → ICL 克隆；带情绪 → M3）→ 硬检查 + 打分 → 挑一个；
拼接整段音频、生成 SRT 字幕与质量报告。输出到 <project>/<script 名>/。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from voicecast.engine import Engine, hard_check, pick

GAP_LINE, GAP_SCENE = 0.35, 1.0
EMO_ZH = {"neutral": "平静", "happy": "开心", "angry": "生气", "sad": "悲伤", "surprised": "惊讶"}


def srt_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round(s % 1 * 1000)):03d}"


def speak(script_path, project, n_best=3, seed=1, engine=None, cer=True, no_emotion=False, progress=None):
    """no_emotion：全部台词按平静生成（选角测试用，排除情绪带来的变化）；报告仍记录原情绪标签。
    progress(比例, 说明)：进度回调（生成占前一半、逐句挑选占后一半）。"""
    eng = engine or Engine()
    progress = progress or (lambda f, d: None)
    project = Path(project)
    cast = yaml.safe_load((project / "cast.yaml").read_text(encoding="utf-8"))
    chars = {c["id"]: c for c in cast["characters"]}
    script = yaml.safe_load(Path(script_path).read_text(encoding="utf-8"))
    out = project / Path(script_path).stem
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    (out / "lines").mkdir(exist_ok=True)

    lines = [{**ln, "scene": sc["id"], "emotion": ln.get("emotion") or "neutral"}
             for sc in script["scenes"] for ln in sc["lines"]]
    for ln in lines:
        if ln["speaker"] not in chars:
            raise ValueError(f"台词说话人 {ln['speaker']} 不在阵容中")
    locks = {i: eng.lock(project / c["ref"], c["ref_text"]) for i, c in chars.items()}
    base = {i: eng.scorer.baseline([str(project / p) for p in c["neutral_probes"]]) for i, c in chars.items()}

    # ① 候选：每个种子整批生成一遍
    sr = None
    for r in range(n_best):
        wavs, sr = eng.say([{"text": ln["text"], "emotion": "neutral" if no_emotion else ln["emotion"],
                             "lock": locks[ln["speaker"]]} for ln in lines], seed + r)
        for k, w in enumerate(wavs):
            sf.write(out / f"candidates/{k:03d}_{r}.wav", w, sr)
        progress(0.5 * (r + 1) / n_best, f"生成第 {r + 1} / {n_best} 轮候选")

    # ② 每句挑选：先过硬检查，再按 音色 ≥ 0.80、清晰度变化 ≥ −1 dB、目标情绪概率 挑
    rows = []
    for k, ln in enumerate(lines):
        progress(0.5 + 0.5 * k / len(lines), f"挑选第 {k + 1} / {len(lines)} 句")
        cs = []
        for r in range(n_best):
            f = out / f"candidates/{k:03d}_{r}.wav"
            probs, _ = hard_check(f, ln["text"])
            target = "neutral" if no_emotion else ln["emotion"]
            cs.append({"r": r, "file": f, "problems": probs, **eng.scorer.score(f, target, base[ln["speaker"]])})
        ok = [c for c in cs if not c["problems"]] or cs
        c = ok[pick(ok, 0.80, -1.0)]
        dst = out / f"lines/{k:03d}_{ln['speaker']}.wav"
        dst.write_bytes(c["file"].read_bytes())
        row = {"idx": k, "scene": ln["scene"], "speaker": ln["speaker"], "name": chars[ln["speaker"]]["name"],
               "emotion": ln["emotion"], "gen_emotion": "neutral" if no_emotion else ln["emotion"],
               "text": ln["text"], "chosen": c["r"],
               "pred": c["pred"], "p_target": round(c["p_target"], 2), "timbre_sim": round(c["timbre_sim"], 3),
               "dcpps": round(c["dcpps"], 2), "problems": "; ".join(c["problems"]), "file": str(dst.relative_to(out))}
        if cer:
            row["cer"] = round(eng.cer(dst, ln["text"]), 3)
        rows.append(row)
        print(f"{k:03d} {row['name']}（{EMO_ZH[ln['emotion']]}）音色 {row['timbre_sim']} 情绪→{row['pred']}", flush=True)
    df = pd.DataFrame(rows)

    # ③ 拼接整段音频 + SRT
    pieces, srt, t = [], [], 0.0
    for k, row in df.iterrows():
        if k > 0:
            gap = GAP_SCENE if row.scene != df.scene[k - 1] else GAP_LINE
            pieces.append(np.zeros(int(gap * sr), dtype=np.float32))
            t += gap
        w, _ = sf.read(out / row.file, dtype="float32")
        pieces.append(w)
        srt.append(f"{k + 1}\n{srt_time(t)} --> {srt_time(t + len(w) / sr)}\n{row['name']}：{row.text}\n")
        t += len(w) / sr
    sf.write(out / "episode.wav", np.concatenate(pieces), sr)
    (out / "episode.srt").write_text("\n".join(srt), encoding="utf-8")
    df.to_csv(out / "lines.csv", index=False)

    # ④ 报告
    warn = df[(df.timbre_sim < 0.80) | (df.dcpps < -2.0) | (df.problems != "") |
              ((df.get("cer", 0) > 0.1) if cer else False)]
    md = [f"# 配音报告：{Path(script_path).name}", "",
          f"阵容最像的一对相似度 {cast['max_pair']}（不做整组挑选时 {cast['max_pair_without_selection']}）；"
          f"撞声对：{cast['collisions'] or '无'}", "",
          "## 阵容", "", pd.DataFrame([{"角色": c["name"], "类型": c["type"], "F0 Hz": c["f0_hz"],
                                         "悲伤清晰度": c["sad_dcpps"], "合格": c["valid"], "提示词": c["prompt"]}
                                        for c in cast["characters"]]).to_markdown(index=False), "",
          "## 逐句", "", df.drop(columns=["file"]).to_markdown(index=False), "",
          f"## 需要注意的句子（音色 < 0.80、清晰度掉超 2 dB、硬检查问题{'、CER > 0.1' if cer else ''}）", "",
          warn[["idx", "name", "emotion", "text", "timbre_sim", "dcpps", "problems"] + (["cer"] if cer else [])]
          .to_markdown(index=False) if len(warn) else "无", "",
          f"总时长 {t:.1f} s；" + ("情绪关闭（全部平静生成）" if no_emotion
                                    else f"情绪识别与目标一致 {(df.pred == df.emotion).mean():.0%}")]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    return df
