"""选角：每个角色 K 个候选 → 设计、锁定、试音、测量 → 整组挑选（最像的一对尽量不像）。
- 角色带声音卡（Agent 填写，voicecast/cards.py；v3 = 类型身份 + 少量修改，v2 = 全字段）：提示词由卡渲染，
  参考句用卡中台词，K 个候选只换种子；类型音高范围硬检查；无合格候选时自动修补再生成一轮
- 角色不带卡（v1）：K 张随机参数卡 × 不同种子，参考句为通用句

输出 <project>/cast.yaml、candidates.csv，以及 voices/<角色>/ref.wav（选中的锁定参考）。
"""
import itertools
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from voicecast import cards as cardv2
from voicecast.engine import LIBRARY, REF_LINE, Engine, hard_check, random_cards, render

PROBES = [("这件事我已经想清楚了，就按我们说的办吧。", None), ("你刚才说什么？我没听清楚，再说一遍好吗？", None),
          ("他走了……再也不会回来了。", "sad")]
COLLIDE = 0.85        # 两个角色的 CAM++ 相似度 ≥ 此值视为撞声（按真实动画阵容定标）
MAX_SAD_DROP = -2.0   # 悲伤试音清晰度下降超过 2 dB 视为不合格（P0.5：约 0.6 dB 听不出，2–3 dB 听得出）
MAX_SAD_DROP_RELAXED = -3.5   # relax_quality 类型（老年、沙哑）：沙哑本是类型特征，放宽
PITCH_W = 0.02        # 挑选时音高差的权重：有效相似度 = CAM++ 相似度 − 0.02 × min(音高差半音, 6)


def select(sims, valid, init):
    """整组挑选：sims[(i, a, j, b)] = 角色 i 的候选 a 与角色 j 的候选 b 的相似度；valid[i] = 可选候选。
    从 init 出发，反复处理最像的一对，换掉其中一人的候选，直到最大相似度不再下降。"""
    choice = dict(init)

    def worst(ch):
        pairs = [(sims[(i, ch[i], j, ch[j])], i, j) for i, j in itertools.combinations(sorted(ch), 2)]
        return max(pairs) if pairs else (0.0, None, None)

    for _ in range(200):
        cur, i, j = worst(choice)
        if i is None:
            break
        best = (cur, None, None)
        for who in (i, j):
            for c in valid[who]:
                if c == choice[who]:
                    continue
                trial = {**choice, who: c}
                w = worst(trial)[0]
                if w < best[0] - 1e-6:
                    best = (w, who, c)
        if best[1] is None:
            break
        choice[best[1]] = best[2]
    return choice


def repair(vtype, card):
    """无合格候选时的自动修补：v3 音高降为低档、去掉语速修改（仍为低档则再去掉音质词）；
    v2 音高降一档（不低于类型下限）、能量降一档、语调改为 steady。"""
    c = dict(card)
    if c.get("version") == 3:
        if c.get("pitch", "lo") == "lo" and not c.get("pace"):
            c.pop("quality", None)
        c["pitch"] = "lo"
        c.pop("pace", None)
        return c
    scale, rng = LIBRARY["pitch_scale"], LIBRARY["types"][vtype]["pitch_range"]
    c["pitch"] = scale[max(rng[0], scale.index(c["pitch"]) - 1)]
    en = LIBRARY["vocab"]["energy"]
    c["energy"] = en[max(0, en.index(c["energy"]) - 1)]
    c["intonation"] = "steady"
    return c


def cast(characters_path, project, k=8, seed=1, engine=None):
    eng = engine or Engine()
    project = Path(project)
    chars = yaml.safe_load(Path(characters_path).read_text(encoding="utf-8"))["characters"]
    for c in chars:
        if c["type"] not in LIBRARY["types"]:
            raise ValueError(f"{c['id']}: 未知类型 {c['type']}")

    def plan(c, card, j0):
        """一个角色的 K 个候选：带卡 = 同一张卡 × K 个种子；不带卡（v1）= K 张随机参数卡。"""
        if card:
            probs = cardv2.check(c["type"], card)
            if probs:
                raise ValueError(f"{c['id']} 声音卡不合格：{probs}")
            items = [(card, cardv2.to_prompt(c["type"], card), card["ref_line"])] * k
        else:
            h = zlib.crc32(c["id"].encode())
            items = [(cd, render(c["type"], cd), REF_LINE) for cd in random_cards(k, seed * 7919 + h)]
        out = []
        for j, (cd, prompt, ref_text) in enumerate(items, start=j0):
            d = project / "voices" / c["id"] / f"cand{j}"
            d.mkdir(parents=True, exist_ok=True)
            out.append({"char": c["id"], "j": j, "type": c["type"], "card": cd, "seed": seed * 1000 + j,
                        "prompt": prompt, "ref_text": ref_text, "dir": d})
        return out

    def generate(cs):
        """设计参考音 → 锁定 → 试音（两句平静 ICL、一句悲伤 M3）→ 测量。"""
        for rt in sorted({x["ref_text"] for x in cs}):
            sub = [x for x in cs if x["ref_text"] == rt]
            ws, sr = eng.design_refs([x["prompt"] for x in sub], [x["seed"] for x in sub], text=rt)
            for x, w in zip(sub, ws):
                sf.write(x["dir"] / "ref.wav", w, sr)
        eng.release_design()
        for x in cs:
            x["lock"] = eng.lock(x["dir"] / "ref.wav", x["ref_text"])
        jobs = [{"text": t, "emotion": e, "lock": x["lock"], "cand": x, "p": p}
                for x in cs for p, (t, e) in enumerate(PROBES)]
        wavs, sr = eng.say(jobs, seed)
        for jb, w in zip(jobs, wavs):
            sf.write(jb["cand"]["dir"] / f"probe{jb['p']}.wav", w, sr)
        for x in cs:
            probs = []
            for p, (t, _) in enumerate(PROBES):
                pr, a = hard_check(x["dir"] / f"probe{p}.wav", t)
                probs += [f"试音{p}:{q}" for q in pr]
                if p == 0:
                    f0 = a["f0_med_hz"]
            neutral = [x["dir"] / f"probe{p}.wav" for p, (_, e) in enumerate(PROBES) if e is None]
            x["base"] = eng.scorer.baseline([str(f) for f in neutral])
            sad = eng.scorer.score(x["dir"] / "probe2.wav", "sad", x["base"])
            t = LIBRARY["types"][x["type"]]
            lo, hi = t["f0_band"]
            if not lo <= f0 <= hi:
                probs.append(f"F0 {f0:.0f} Hz 超出类型范围 {lo}–{hi}")
            floor = MAX_SAD_DROP_RELAXED if t.get("relax_quality") else MAX_SAD_DROP
            x.update(f0=f0, sad_dcpps=sad["dcpps"], sad_ok=sad["pred"] == "sad", problems=probs,
                     valid=not probs and sad["dcpps"] >= floor)

    # ① 候选 + ② 生成与测量
    cands = [x for c in chars for x in plan(c, c.get("card"), 0)]
    generate(cands)
    # ③ 自动修补：带卡角色若无合格候选，修补卡片再生成一轮
    repaired = {}
    for c in chars:
        if c.get("card") and not any(x["valid"] for x in cands if x["char"] == c["id"]):
            repaired[c["id"]] = repair(c["type"], c["card"])
    if repaired:
        extra = [x for c in chars if c["id"] in repaired for x in plan(c, repaired[c["id"]], k)]
        generate(extra)
        cands += extra
    rows = [{"char": x["char"], "cand": x["j"], "seed": x["seed"], "f0_hz": round(x["f0"], 1),
             "sad_dcpps": round(x["sad_dcpps"], 2), "sad_recognized": x["sad_ok"], "valid": x["valid"],
             "repaired": x["char"] in repaired and x["j"] >= k, "problems": "; ".join(x["problems"]),
             "prompt": x["prompt"]} for x in cands]

    # ④ 整组挑选：只在合格候选中挑；仍无合格候选的角色取 F0 最接近类型范围的候选并标出
    ids = [c["id"] for c in chars]
    by = {i: {x["j"]: x for x in cands if x["char"] == i} for i in ids}
    sims, cam = {}, {}
    for i, j in itertools.combinations(ids, 2):
        for a in by[i].values():
            for b in by[j].values():
                c_ = float(a["base"][0] @ b["base"][0])
                dst = min(abs(12 * np.log2(a["f0"] / b["f0"])), 6.0)
                cam[(i, a["j"], j, b["j"])] = cam[(j, b["j"], i, a["j"])] = c_
                sims[(i, a["j"], j, b["j"])] = sims[(j, b["j"], i, a["j"])] = c_ - PITCH_W * dst

    def band_gap(x):
        lo, hi = LIBRARY["types"][x["type"]]["f0_band"]
        return max(lo - x["f0"], x["f0"] - hi, 0)

    no_valid = [i for i in ids if not any(x["valid"] for x in by[i].values())]
    valid = {i: [j for j, x in by[i].items() if x["valid"]] or [min(by[i], key=lambda j: band_gap(by[i][j]))]
             for i in ids}
    init = {i: max(valid[i], key=lambda c: by[i][c]["sad_dcpps"]) for i in ids}
    choice = select(sims, valid, init)

    # ⑤ 输出
    out = {"characters": [], "collide_threshold": COLLIDE, "k": k, "seed": seed,
           "repaired": list(repaired), "no_valid_candidate": no_valid}
    for c in chars:
        x = by[c["id"]][choice[c["id"]]]
        vdir = project / "voices" / c["id"]
        for f in ["ref.wav", "probe0.wav", "probe1.wav", "probe2.wav"]:
            shutil.copy(x["dir"] / f, vdir / f)
        out["characters"].append({**c, "card": x["card"], "prompt": x["prompt"], "seed": x["seed"],
                                  "candidate": x["j"], "ref": f"voices/{c['id']}/ref.wav", "ref_text": x["ref_text"],
                                  "neutral_probes": [f"voices/{c['id']}/probe0.wav", f"voices/{c['id']}/probe1.wav"],
                                  "f0_hz": round(x["f0"], 1), "sad_dcpps": round(x["sad_dcpps"], 2),
                                  "valid": x["valid"], "problems": x["problems"]})
    M = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for i in ids:
        for j in ids:
            M.loc[i, j] = 1.0 if i == j else cam[(i, choice[i], j, choice[j])]
    out["max_pair"] = round(float(M.where(~np.eye(len(ids), dtype=bool)).max().max()), 3)
    out["collisions"] = [[i, j, round(float(M.loc[i, j]), 3)] for i, j in itertools.combinations(ids, 2)
                         if M.loc[i, j] >= COLLIDE]
    # 对照：不做整组挑选（每人取初始候选）时的最大相似度
    out["max_pair_without_selection"] = round(max(cam[(i, init[i], j, init[j])]
                                                  for i, j in itertools.combinations(ids, 2)), 3)
    (project / "cast.yaml").write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(project / "candidates.csv", index=False)
    M.round(3).to_csv(project / "cast_similarity.csv")
    return out
