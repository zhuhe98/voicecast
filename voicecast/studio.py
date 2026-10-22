"""选角工作台：每个角色给出 3 个合格、互不撞的选项（音色库 + 按声音卡现场生成），人选择 / 换一批 / 锁定，生成阵容试音。

用法：python -m voicecast.studio 角色表.yaml --project 项目目录 [--port 7873] [--share [--password PW]] [--dtype bf16] [--batch 4]
      （VSCode 转发端口后打开 http://127.0.0.1:7873；或 --share 生成带密码的公开链接）
"""
import argparse, itertools, secrets, shutil
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from voicecast import bank as vbank
from voicecast import cards as vcards
from voicecast.backend.models import add_engine_args
from voicecast.cast import COLLIDE, PITCH_W, select
from voicecast.engine import LIBRARY, REF_LINE, ROOT, Engine, hard_check

N_OPT, POOL_BANK, FRESH_K, MAX_CHARS = 3, 16, 4, 10


def sim(a, b):
    """有效相似度：CAM++ 相似度 − 音高差惩罚（与 cast.py 一致）。"""
    return float(a["emb"] @ b["emb"]) - PITCH_W * min(abs(12 * np.log2(a["f0"] / b["f0"])), 6.0)


class Studio:
    def __init__(self, characters_path, project, bank="bank", engine=None):
        self.chars = yaml.safe_load(Path(characters_path).read_text(encoding="utf-8"))["characters"][:MAX_CHARS]
        self.project = Path(project)
        self.project.mkdir(parents=True, exist_ok=True)
        self.bank = bank
        self.eng = engine or Engine()
        self.pool = {c["id"]: [] for c in self.chars}     # 候选
        self.shown = {c["id"]: [] for c in self.chars}    # 当前展示的 3 个（候选下标）
        self.seen = {c["id"]: set() for c in self.chars}  # 展示过的
        self.pick = {}                                     # 当前选择（候选下标）

    # ── 候选 ──────────────────────────────────────────
    def _bank_pool(self, c):
        idx, E = vbank.load(self.bank)
        if not len(idx):
            return []
        used = pd.read_csv(ROOT / self.bank / "usage.csv") if (ROOT / self.bank / "usage.csv").exists() else pd.DataFrame(columns=["id", "project"])
        other = set(used[used.project != str(self.project)].id)
        m = idx[(idx.type == c["type"]) & idx.ok & ~idx.id.isin(other)].copy()
        m = m.sort_values("proto_sim", ascending=False, na_position="last").head(POOL_BANK)
        return [{"source": "音色库", "key": r.id, "wav": str(ROOT / self.bank / r.file), "prompt": r.prompt,
                 "ref_text": r.ref_text, "f0": r.f0_hz, "emb": E[i], "proto_sim": r.proto_sim} for i, r in m.iterrows()]

    def _fresh(self, cs, k_start=0, progress=None):
        """按声音卡现场生成 FRESH_K 个候选（角色须带 v2 / v3 卡）。progress(比例, 说明) 覆盖本阶段 0–1。"""
        progress = progress or (lambda f, d: None)
        jobs = []
        for c in cs:
            card = c.get("card")
            if not card or vcards.check(c["type"], card):
                continue
            for k in range(k_start, k_start + FRESH_K):
                d = self.project / "fresh" / c["id"]
                d.mkdir(parents=True, exist_ok=True)
                jobs.append({"c": c, "k": k, "prompt": vcards.to_prompt(c["type"], card), "ref_text": card["ref_line"],
                             "wav": str(d / f"{k}.wav"), "seed": 500 + k_start})   # 同批同种子，批内各项仍各不相同
        if not jobs:
            return
        progress(0.0, "加载声音设计模型……")
        out, sr = self.eng.design_refs([j["prompt"] for j in jobs], [j["seed"] for j in jobs],
                                       text=[j["ref_text"] for j in jobs],
                                       on_batch=lambda d, t: progress(0.85 * d / t, f"按人设生成候选 {d} / {t}"))
        progress(0.9, "测量候选（音高、说话人向量）……")
        for j, w in zip(jobs, out):
            sf.write(j["wav"], w, sr)
        self.eng.release_design()
        from voicecast.backend.spk import embed
        E = embed(self.eng.scorer.cam, [j["wav"] for j in jobs])
        for j, e in zip(jobs, E):
            probs, a = hard_check(j["wav"], j["ref_text"])
            lo, hi = LIBRARY["types"][j["c"]["type"]]["f0_band"]
            if not lo <= a["f0_med_hz"] <= hi:
                probs.append("F0 超范围")
            if probs:
                continue
            self.pool[j["c"]["id"]].append({"source": "按人设生成", "key": f"fresh:{j['c']['id']}:{j['k']}",
                                            "wav": j["wav"], "prompt": j["prompt"], "ref_text": j["ref_text"],
                                            "f0": a["f0_med_hz"], "emb": e / np.linalg.norm(e),
                                            "proto_sim": vbank.proto_scores(j["c"]["type"], e / np.linalg.norm(e))[0]})

    # ── 选项 ──────────────────────────────────────────
    def _rank(self, cid):
        """该角色的候选按"与其他角色当前选择的最大有效相似度"升序；剔除撞声的。"""
        others = [self.pool[o][self.pick[o]] for o in self.pick if o != cid]
        rows = []
        for i, x in enumerate(self.pool[cid]):
            worst = max((sim(x, y) for y in others), default=0.0)
            rows.append((worst, i))
        return [i for w, i in sorted(rows) if w < COLLIDE] or [i for _, i in sorted(rows)]

    def _fill(self, cid, keep_pick=True):
        rank = [i for i in self._rank(cid) if i not in self.seen[cid]]
        cur = [self.pick[cid]] if keep_pick and cid in self.pick else []
        out = list(cur)
        for src in ("按人设生成", "音色库"):   # 尽量两个来源都有
            for i in rank:
                if i not in out and self.pool[cid][i]["source"] == src:
                    out.append(i)
                    break
        out += [i for i in rank if i not in out][: N_OPT - len(out)]
        self.shown[cid] = out[:N_OPT]
        self.seen[cid] |= set(self.shown[cid])

    def propose(self, progress=None):
        progress = progress or (lambda f, d: None)
        progress(0.02, "检索音色库……")
        for c in self.chars:
            self.pool[c["id"]] = self._bank_pool(c)
        self._fresh(self.chars, progress=lambda f, d: progress(0.05 + 0.85 * f, d))
        progress(0.93, "整组挑选（让角色彼此不撞）……")
        ids = [c["id"] for c in self.chars if self.pool[c["id"]]]
        sims = {(i, a, j, b): sim(self.pool[i][a], self.pool[j][b])
                for i, j in itertools.permutations(ids, 2)
                for a in range(len(self.pool[i])) for b in range(len(self.pool[j]))}
        init = {i: next((k for k, x in enumerate(self.pool[i]) if x["source"] == "按人设生成"), 0) for i in ids}
        self.pick = select(sims, {i: list(range(len(self.pool[i]))) for i in ids}, init)
        for i in ids:
            self.seen[i] = set()
            self._fill(i)

    def reroll(self, cid, progress=None):
        if len([i for i in range(len(self.pool[cid])) if i not in self.seen[cid]]) < N_OPT:
            c = next(x for x in self.chars if x["id"] == cid)
            ks = [int(x["key"].split(":")[-1]) for x in self.pool[cid] if x["key"].startswith("fresh:")]
            self._fresh([c], k_start=max(ks, default=-1) + 1, progress=progress)
        self._fill(cid, keep_pick=False)

    # ── 锁定 ──────────────────────────────────────────
    def lock(self, progress=None):
        progress = progress or (lambda f, d: None)
        progress(0.05, "锁定声音（加载克隆模型）……")
        out = {"characters": [], "collide_threshold": COLLIDE, "source": "studio"}
        jobs = []
        for c in self.chars:
            x = self.pool[c["id"]][self.pick[c["id"]]]
            vdir = self.project / "voices" / c["id"]
            vdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(x["wav"], vdir / "ref.wav")
            lock = self.eng.lock(vdir / "ref.wav", x["ref_text"])
            jobs += [{"text": t, "emotion": None, "lock": lock, "cid": c["id"], "p": p}
                     for p, t in enumerate(["这件事我已经想清楚了，就按我们说的办吧。", "你刚才说什么？我没听清楚，再说一遍好吗？"])]
            out["characters"].append({**c, "prompt": x["prompt"], "source": x["source"], "voice_key": x["key"],
                                      "ref": f"voices/{c['id']}/ref.wav", "ref_text": x["ref_text"], "f0_hz": round(float(x["f0"]), 1),
                                      "sad_dcpps": None, "valid": True,
                                      "neutral_probes": [f"voices/{c['id']}/probe0.wav", f"voices/{c['id']}/probe1.wav"]})
        progress(0.5, "生成锁定后的试音……")
        wavs, sr = self.eng.say(jobs, 1)
        for j, w in zip(jobs, wavs):
            sf.write(self.project / "voices" / j["cid"] / f"probe{j['p']}.wav", w, sr)
        progress(1.0, "锁定完成")
        picks = [self.pool[c["id"]][self.pick[c["id"]]] for c in self.chars]
        pairs = [(float(a["emb"] @ b["emb"]), ca["id"], cb["id"]) for (ca, a), (cb, b)
                 in itertools.combinations(zip(self.chars, picks), 2)]
        out["max_pair"] = round(max(pairs)[0], 3) if pairs else 0.0
        out["max_pair_without_selection"] = out["max_pair"]
        out["collisions"] = [[i, j, round(s, 3)] for s, i, j in pairs if s >= COLLIDE]
        (self.project / "cast.yaml").write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
        use = pd.DataFrame([{"id": x["key"], "project": str(self.project)} for x in picks if x["source"] == "音色库"])
        f = ROOT / self.bank / "usage.csv"
        if len(use):
            use.to_csv(f, mode="a", header=not f.exists(), index=False)
        from voicecast.export import voices
        voices(self.project)
        return out

    def lineup(self, progress=None):
        from voicecast.speak import speak
        script = self.project / "lineup.yaml"
        script.write_text(yaml.safe_dump({"scenes": [{"id": "lineup", "title": "阵容试音", "lines": [
            {"speaker": c["id"], "text": c["intro"]} for c in self.chars if c.get("intro")]}]}, allow_unicode=True), encoding="utf-8")
        res = speak(script, self.project, n_best=3, seed=1, engine=self.eng, cer=False, no_emotion=True,
                    progress=progress)
        from voicecast.export import voices
        voices(self.project)   # 带上试听样本重新导出
        return res


# ── 页面 ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(prog="voicecast.studio")
    ap.add_argument("characters")
    ap.add_argument("--project", required=True)
    ap.add_argument("--bank", default="bank")
    ap.add_argument("--port", type=int, default=7873)
    ap.add_argument("--share", action="store_true", help="Gradio 公开链接（带密码）")
    ap.add_argument("--password")
    add_engine_args(ap)
    a = ap.parse_args()
    st = Studio(a.characters, a.project, a.bank, engine=Engine(a.dtype, a.batch))
    chars = st.chars

    def opt_md(x, n):
        ps = "" if np.isnan(x.get("proto_sim", np.nan)) else f"　类型相似 {x['proto_sim']:.2f}"
        return f"**选项 {n}**　{x['source']}　F0 {x['f0']:.0f} Hz{ps}\n\n<small>`{x['prompt']}`</small>"

    def render():
        outs = []
        for k in range(MAX_CHARS):
            if k >= len(chars) or not st.pool[chars[k]["id"]]:
                outs += [gr.update(visible=False)] + [None, ""] * N_OPT + [gr.update(choices=[], value=None)]
                continue
            cid = chars[k]["id"]
            outs.append(gr.update(visible=True))
            labels = []
            for n in range(N_OPT):
                if n < len(st.shown[cid]):
                    x = st.pool[cid][st.shown[cid][n]]
                    outs += [x["wav"], opt_md(x, n + 1)]
                    labels.append(f"选项 {n + 1}")
                else:
                    outs += [None, ""]
            cur = st.shown[cid].index(st.pick[cid]) if st.pick.get(cid) in st.shown[cid] else None
            outs.append(gr.update(choices=labels, value=None if cur is None else labels[cur]))
        return outs

    with gr.Blocks(title="选角工作台") as demo:
        gr.Markdown(f"# 选角工作台　`{st.project.name}`\n每个角色 3 个选项（音色库 + 按人设生成），已保证类型音高合格、彼此不撞。"
                    "选一个；不满意点\"换一批\"（会避开其他角色当前选的声音）。\n\n"
                    "生成选项约 3–4 分钟、锁定并生成阵容试音约 2 分钟，进度显示在下方状态栏。")
        status = gr.Markdown()
        start = gr.Button("① 生成选项", variant="primary")
        slots = []
        for k in range(MAX_CHARS):
            c = chars[k] if k < len(chars) else {"name": "", "type": "loli", "description": ""}
            with gr.Group(visible=False) as grp:
                gr.Markdown(f"### {c['name']}　<small>{LIBRARY['types'][c['type']]['zh']}｜{c.get('description', '')}</small>")
                with gr.Row():
                    cells = []
                    for n in range(N_OPT):
                        with gr.Column():
                            cells += [gr.Audio(show_label=False), gr.Markdown()]
                with gr.Row():
                    radio = gr.Radio([], label="选择")
                    rr = gr.Button("换一批", size="sm")
            slots.append((grp, cells, radio, rr))
        lockb = gr.Button("② 锁定并生成阵容试音", variant="primary")
        lineup_audio = gr.Audio(label="阵容试音（依次自我介绍）")
        lineup_md = gr.Markdown()
        flat = [x for grp, cells, radio, _ in slots for x in [grp, *cells, radio]]

        def do_start(progress=gr.Progress()):
            st.propose(progress=lambda f, d: progress(f, desc=d))
            return ["选项已生成。"] + render()

        start.click(do_start, None, [status] + flat)
        for k, (grp, cells, radio, rr) in enumerate(slots):
            def choose(v, k=k):
                if k < len(chars) and v:
                    cid = chars[k]["id"]
                    st.pick[cid] = st.shown[cid][int(v.split()[-1]) - 1]
                return f"已选：{chars[k]['name']} {v}" if k < len(chars) else ""

            def reroll(k=k, progress=gr.Progress()):
                st.reroll(chars[k]["id"], progress=lambda f, d: progress(f, desc=d))
                return [f"{chars[k]['name']} 已换一批。"] + render()

            radio.change(choose, radio, status)
            rr.click(reroll, None, [status] + flat)

        def do_lock(progress=gr.Progress()):
            out = st.lock(progress=lambda f, d: progress(0.3 * f, desc=d))
            df = st.lineup(progress=lambda f, d: progress(0.3 + 0.7 * f, desc=f"阵容试音：{d}"))
            md = f"阵容最像的一对相似度 **{out['max_pair']}**　撞声：{out['collisions'] or '无'}\n\n" + "\n".join(
                f"- **{r['name']}**：{r.text}" for _, r in df.iterrows())
            return str(st.project / "lineup" / "episode.wav"), md, "已锁定。声音已导出（可拷贝）：" + str(st.project / "export") + "（另有 export.zip）"

        lockb.click(do_lock, None, [lineup_audio, lineup_md, status])

    auth = None
    if a.share:
        password = a.password or secrets.token_urlsafe(8)
        print(f"用户名 rater　密码 {password}")
        auth = ("rater", password)
    demo.launch(server_name="127.0.0.1", server_port=a.port, allowed_paths=[str(st.project), str(ROOT / a.bank)],
                share=a.share, auth=auth)


if __name__ == "__main__":
    main()
