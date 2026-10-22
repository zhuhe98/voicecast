"""音色库：离线按类型批量设计声音 → 硬检查（类型音高范围、时长、静音、爆音）→ 入库（音频 + 说话人向量 + 元数据）。
选角时从库中检索，配合现场按声音卡生成的候选（voicecast/studio.py）。

用法：python -m voicecast.bank build [--n 24] [--types loli,genki_girl] [--bank bank] [--dtype bf16] [--batch 4]
库结构：<bank>/<类型>/<id>.wav、<bank>/index.csv、<bank>/emb_campplus.npy（与 index 行对齐）、<bank>/usage.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from voicecast.backend.models import add_engine_args
from voicecast.engine import LIBRARY, REF_LINE, ROOT, Engine, hard_check
from voicecast.library import type_prompt

PROTO = np.load(Path(__file__).parent / "prototypes_campplus.npz")
PROTOS = dict(zip(PROTO["types"], PROTO["vectors"]))


def proto_scores(vtype, e):
    """与本类型原型的相似度，以及与同性别其他类型原型的最大相似度（无原型时为 NaN）。"""
    g = LIBRARY["types"][vtype]["gender"]
    own = float(e @ PROTOS[vtype]) if vtype in PROTOS else np.nan
    others = [float(e @ v) for t, v in PROTOS.items() if t != vtype and LIBRARY["types"][t]["gender"] == g]
    return own, max(others) if others else np.nan


def _check(args):
    path, vtype = args
    probs, a = hard_check(path, REF_LINE)
    lo, hi = LIBRARY["types"][vtype]["f0_band"]
    if not lo <= a["f0_med_hz"] <= hi:
        probs.append(f"F0 {a['f0_med_hz']:.0f} 超出 {lo}–{hi}")
    return a["f0_med_hz"], a["hnr"], probs


def load(bank="bank"):
    bank = ROOT / bank
    if not (bank / "index.csv").exists():
        return pd.DataFrame(), np.zeros((0, 192), dtype=np.float32)
    return pd.read_csv(bank / "index.csv"), np.load(bank / "emb_campplus.npy")


def build(types, n, bank="bank", seed0=100, engine=None):
    eng = engine or Engine()
    root = ROOT / bank
    idx, E = load(bank)
    rows = []
    for t in types:
        (root / t).mkdir(parents=True, exist_ok=True)
        start = int((idx.type == t).sum()) if len(idx) else 0
        for k in range(start, start + n):
            lv = "lo" if k % 2 == 0 else "hi"
            rows.append({"id": f"{t}_{k:03d}", "type": t, "level": lv, "seed": seed0 + k, "prompt": type_prompt(t, lv)})
    wavs, sr = eng.design_refs([r["prompt"] for r in rows], [r["seed"] for r in rows], text=REF_LINE)
    for r, w in zip(rows, wavs):
        r["file"] = f"{r['type']}/{r['id']}.wav"
        sf.write(root / r["file"], w, sr)
    eng.release_design()
    with ProcessPoolExecutor(16) as ex:
        checks = list(ex.map(_check, [(root / r["file"], r["type"]) for r in rows]))
    from voicecast.backend.spk import embed
    newE = embed(eng.scorer.cam, [root / r["file"] for r in rows])
    for r, (f0, hnr, probs), e in zip(rows, checks, newE):
        own, other = proto_scores(r["type"], e)
        r.update(f0_hz=round(f0, 1), hnr=round(hnr, 2), proto_sim=round(own, 3), proto_margin=round(own - other, 3),
                 ok=not probs, problems="; ".join(probs), ref_text=REF_LINE)
    idx = pd.concat([idx, pd.DataFrame(rows)], ignore_index=True)
    E = np.concatenate([E, newE.astype(np.float32)]) if len(E) else newE.astype(np.float32)
    idx.to_csv(root / "index.csv", index=False)
    np.save(root / "emb_campplus.npy", E)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(prog="voicecast.bank")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=24, help="每个类型新增几个声音")
    b.add_argument("--types", help="逗号分隔；默认全部类型")
    b.add_argument("--bank", default="bank")
    add_engine_args(b)
    a = ap.parse_args()
    types = a.types.split(",") if a.types else list(LIBRARY["types"])
    df = build(types, a.n, a.bank, engine=Engine(a.dtype, a.batch))
    s = df.groupby("type").agg(入库=("ok", "sum"), 生成=("ok", "size"), F0=("f0_hz", "median"), 原型相似=("proto_sim", "median"))
    print(s.to_string())
    print(f"合格 {int(df.ok.sum())} / {len(df)}")


if __name__ == "__main__":
    main()
