"""两种输出：
- prompts：只输出每个角色的声音描述（英文提示词 + 参考台词 + 音高范围），给其他模型或其他用途；不加载模型、不需要显卡。
- voices：把选定并锁定的声音导出为可拷贝的文件夹（参考录音 + 对应文字 + 试听样本 + 清单），可用于任何支持参考音频克隆的 TTS。
"""
import itertools
import shutil
from pathlib import Path

import yaml

from voicecast import cards
from voicecast.library import LIBRARY, type_prompt

PACE_DEFAULT = "type"


def _md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |\n" for r in rows)


def prompts(characters_path, out):
    """角色表 → <out>/prompts.yaml 与 prompts.md。声音卡不合格时报错（列出问题）。"""
    chars = yaml.safe_load(Path(characters_path).read_text(encoding="utf-8"))["characters"]
    rows, errors = [], []
    for c in chars:
        t = LIBRARY["types"].get(c["type"])
        if t is None:
            errors.append(f"{c['name']}：未知类型 {c['type']}")
            continue
        card = c.get("card")
        if card:
            probs = cards.check(c["type"], card)
            if probs:
                errors.append(f"{c['name']}：" + "；".join(probs))
                continue
        lo, hi = t["f0_band"]
        rows.append({"id": c["id"], "name": c["name"], "type": t["zh"], "description": c.get("description", ""),
                     "prompt": cards.to_prompt(c["type"], card) if card else type_prompt(c["type"]),
                     "ref_line": (card or {}).get("ref_line", ""), "pitch_hz": f"{lo}–{hi}",
                     "_type": c["type"], "_level": (card or {}).get("pitch", "lo"), "_pace": (card or {}).get("pace", PACE_DEFAULT)})
    if errors:
        raise ValueError("声音卡有问题：\n" + "\n".join(errors))

    # 文字层面的撞声提醒：同类型且音高档、语速都相同的两个角色
    warnings = [f"{a['name']} 与 {b['name']} 同为{a['type']}，音高档和语速都相同，生成后容易像同一个人；建议一高一低或换语速，并试听确认"
                for a, b in itertools.combinations(rows, 2)
                if a["_type"] == b["_type"] and a["_level"] == b["_level"] and a["_pace"] == b["_pace"]]
    for r in rows:
        for k in ("_type", "_level", "_pace"):
            r.pop(k)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.yaml").write_text(yaml.safe_dump({"characters": rows, "warnings": warnings},
                                                     allow_unicode=True, sort_keys=False), encoding="utf-8")
    md = ["# 角色声音描述\n\n",
          "每个角色一段英文声音描述（prompt），可直接用于支持文字描述生成声音的 TTS 模型；"
          "`ref_line` 是建议的试音台词（语气平稳的陈述句）；`pitch_hz` 是该类型试音音高中位数的合理范围，可用来检查生成结果的年龄性别是否跑偏。\n\n",
          _md_table(rows, ["name", "type", "description", "prompt", "ref_line", "pitch_hz"])]
    if warnings:
        md += ["\n## 可能撞声的组合\n"] + [f"- {w}\n" for w in warnings]
    md.append("\n## 使用提示\n\n- 同一段描述，换随机种子生成的声音差别很大：每个角色多生成几个再挑。\n"
              "- 只靠文字无法保证角色之间分得清；生成后请把同类型、相邻类型的角色放在一起试听。"
              "需要自动保证不撞声时，用 voicecast 的声音输出模式。\n")
    (out / "prompts.md").write_text("".join(md), encoding="utf-8")
    return rows, warnings


def voices(project, out=None):
    """锁定后的项目（cast.yaml）→ <out>/<序号_角色名>/{reference.wav, reference.txt, sample.wav}、voices.yaml、使用说明.txt。"""
    project = Path(project)
    cast = yaml.safe_load((project / "cast.yaml").read_text(encoding="utf-8"))
    out = Path(out) if out else project / "export"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    samples = {}
    lines = project / "lineup" / "lines.csv"
    if lines.exists():
        import pandas as pd
        samples = {r.speaker: project / "lineup" / r.file for r in pd.read_csv(lines).itertuples()}
    manifest = []
    for k, c in enumerate(cast["characters"], 1):
        d = out / f"{k:02d}_{c['name']}"
        d.mkdir()
        shutil.copy(project / c["ref"], d / "reference.wav")
        (d / "reference.txt").write_text(c["ref_text"] + "\n", encoding="utf-8")
        sample = samples.get(c["id"]) or next((project / p for p in c.get("neutral_probes", []) if (project / p).exists()), None)
        if sample:
            shutil.copy(sample, d / "sample.wav")
        manifest.append({"id": c["id"], "name": c["name"], "type": LIBRARY["types"][c["type"]]["zh"],
                         "description": c.get("description", ""), "folder": d.name,
                         "reference_audio": f"{d.name}/reference.wav", "reference_text": c["ref_text"],
                         "sample": f"{d.name}/sample.wav" if sample else None,
                         "prompt": c.get("prompt", ""), "pitch_hz": c.get("f0_hz")})
    (out / "voices.yaml").write_text(yaml.safe_dump({"characters": manifest, "max_pair_similarity": cast.get("max_pair")},
                                                    allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "使用说明.txt").write_text(
        "每个文件夹是一个角色的声音：\n"
        "  reference.wav  参考录音（锁定的声音）\n"
        "  reference.txt  参考录音对应的文字\n"
        "  sample.wav     试听样本\n\n"
        "在任何支持参考音频克隆的 TTS（如 Qwen3-TTS Base 等零样本克隆模型）中，把 reference.wav 和 reference.txt "
        "作为参考，即可让该角色说任意台词。voices.yaml 汇总了全部角色、生成这个声音所用的描述（prompt）和音高。\n"
        "这些声音由模型按文字描述合成，不是任何真人的声音。\n", encoding="utf-8")
    shutil.make_archive(str(out), "zip", out.parent, out.name)
    return out
