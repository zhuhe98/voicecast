"""审听页：阵容（每角色试音）+ 整段音频 + 逐句（台词、情绪、测量、音频）。

用法：python -m voicecast.review <project> [--script script] [--port 7866] [--share]
"""
import argparse, secrets
from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

from voicecast.engine import LIBRARY
from voicecast.speak import EMO_ZH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--script", default="script", help="台词表名（project 下的子目录）")
    ap.add_argument("--port", type=int, default=7866)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--password")
    args = ap.parse_args()
    proj = Path(args.project).resolve()
    cast = yaml.safe_load((proj / "cast.yaml").read_text(encoding="utf-8"))
    sdir = proj / args.script
    lines = pd.read_csv(sdir / "lines.csv")
    chars = cast["characters"]

    with gr.Blocks(title="配音审听") as demo:
        gr.Markdown(f"# 配音审听　`{proj.name}`\n阵容最像的一对相似度 **{cast['max_pair']}**"
                    f"（不做整组挑选时 {cast['max_pair_without_selection']}；≥ {cast['collide_threshold']} 视为撞声）"
                    f"　撞声：{cast['collisions'] or '无'}")
        with gr.Tab("阵容"):
            for i in range(0, len(chars), 4):
                with gr.Row():
                    for c in chars[i:i + 4]:
                        with gr.Column(min_width=220):
                            t = LIBRARY["types"][c["type"]]
                            gr.Markdown(f"### {c['name']}\n**{t['zh']}**　F0 {c['f0_hz']:.0f} Hz\n\n"
                                        f"<small>{c.get('description', '')}</small>\n\n<small>`{c['prompt']}`</small>")
                            gr.Audio(str(proj / c["neutral_probes"][0]), label="试音：平静")
                            gr.Audio(str(proj / f"voices/{c['id']}/probe2.wav"), label="试音：悲伤")
        with gr.Tab("整段"):
            gr.Audio(str(sdir / "episode.wav"), label="整段音频")
            gr.Markdown(f"字幕：`{sdir / 'episode.srt'}`")
        with gr.Tab("逐句"):
            gr.Markdown("✓/✗ = 情绪识别是否与目标一致（平静句常被误判为开心）；音色 = 与该角色试音的相似度；"
                        "清晰度 = 相对该角色平静试音（dB）；CER = 读字错误率（含同音字 / 生僻字误识别）")
            for _, r in lines.iterrows():
                gen = r.get("gen_emotion", r.emotion)
                ok = "✓" if r.pred == gen else f"✗ 识别为{EMO_ZH.get(r.pred, r.pred)}"
                warn = " ⚠" if (r.timbre_sim < 0.80 or r.dcpps < -2 or (r.get("cer", 0) or 0) > 0.1) else ""
                with gr.Row():
                    tag = EMO_ZH[r.emotion] + ("，生成时关闭情绪" if gen != r.emotion else "")
                    gr.Markdown(f"**{r.idx + 1}. {r['name']}**（{tag}）{warn}\n\n{r.text}\n\n"
                                f"<small>{ok}　音色 {r.timbre_sim:.2f}　清晰度 {r.dcpps:+.1f}"
                                + (f"　CER {r.cer:.2f}" if "cer" in r else "") + "</small>")
                    gr.Audio(str(sdir / r.file), show_label=False)

    auth = None
    if args.share:
        password = args.password or secrets.token_urlsafe(8)
        print(f"用户名 rater　密码 {password}")
        auth = ("rater", password)
    demo.launch(server_name="127.0.0.1", server_port=args.port, allowed_paths=[str(proj)], share=args.share, auth=auth)


if __name__ == "__main__":
    main()
