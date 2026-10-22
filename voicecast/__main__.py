"""命令行：
  python -m voicecast prompts 角色表.yaml --out 目录                       # 只输出声音描述（不需要显卡）
  python -m voicecast export --project 项目目录 [--out 目录]               # 导出锁定的声音（参考录音 + 文字 + 样本）
  python -m voicecast cast  角色表.yaml --project 项目目录 [--k 8] [--seed 1]
  python -m voicecast speak 台词表.yaml --project 项目目录 [--n-best 3] [--seed 1] [--no-cer] [--no-emotion]
  python -m voicecast run   角色表.yaml 台词表.yaml --project 项目目录      # 选角 + 配音
  python -m voicecast lineup 角色表.yaml --project 项目目录                # 选角 + 阵容试音（角色表 intro 字段）
需要显卡的命令（cast / speak / run / lineup）都可加 --dtype bf16 --batch 4（或环境变量 VOICECAST_DTYPE / VOICECAST_BATCH）
"""
import argparse
from pathlib import Path

import yaml

from voicecast.backend.models import add_engine_args


def main():
    ap = argparse.ArgumentParser(prog="voicecast")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cast")
    c.add_argument("characters")
    s = sub.add_parser("speak")
    s.add_argument("script")
    r = sub.add_parser("run")
    r.add_argument("characters")
    r.add_argument("script")
    ln = sub.add_parser("lineup")
    ln.add_argument("characters")
    pr = sub.add_parser("prompts", help="只输出每个角色的声音描述（不加载模型）")
    pr.add_argument("characters")
    pr.add_argument("--out", required=True)
    ex = sub.add_parser("export", help="导出锁定的声音为可拷贝的文件夹")
    ex.add_argument("--project", required=True)
    ex.add_argument("--out")
    for p in (c, s, r, ln):
        p.add_argument("--project", required=True)
        p.add_argument("--seed", type=int, default=1)
        add_engine_args(p)
    for p in (c, r, ln):
        p.add_argument("--k", type=int, default=8, help="每个角色的候选数")
    for p in (s, r):
        p.add_argument("--n-best", type=int, default=3, help="每句生成几个候选")
        p.add_argument("--no-cer", action="store_true")
        p.add_argument("--no-emotion", action="store_true", help="全部台词按平静生成（选角测试用）")
    a = ap.parse_args()

    if a.cmd == "prompts":
        from voicecast.export import prompts
        try:
            rows, warnings = prompts(a.characters, a.out)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"已输出 {len(rows)} 个角色的声音描述：{Path(a.out) / 'prompts.md'}")
        for w in warnings:
            print("提醒：", w)
        return
    if a.cmd == "export":
        from voicecast.export import voices
        print("已导出声音：", voices(a.project, a.out))
        return

    from voicecast.engine import Engine
    eng = Engine(a.dtype, a.batch)
    if a.cmd == "lineup":
        from voicecast.cast import cast
        from voicecast.speak import speak
        out = cast(a.characters, a.project, k=a.k, seed=a.seed, engine=eng)
        print(f"阵容完成：最像的一对 {out['max_pair']}（不挑选 {out['max_pair_without_selection']}），撞声 {out['collisions'] or '无'}")
        chars = yaml.safe_load(Path(a.characters).read_text(encoding="utf-8"))["characters"]
        script = Path(a.project) / "lineup.yaml"
        script.write_text(yaml.safe_dump({"scenes": [{"id": "lineup", "title": "阵容试音", "lines": [
            {"speaker": c["id"], "text": c["intro"]} for c in chars if c.get("intro")]}]}, allow_unicode=True), encoding="utf-8")
        speak(script, a.project, n_best=3, seed=a.seed, engine=eng, no_emotion=True)
        from voicecast.export import voices
        print("阵容试音完成：", a.project, "；声音已导出：", voices(a.project))
        return
    if a.cmd in ("cast", "run"):
        from voicecast.cast import cast
        out = cast(a.characters, a.project, k=a.k, seed=a.seed, engine=eng)
        print(f"阵容完成：最像的一对 {out['max_pair']}（不挑选 {out['max_pair_without_selection']}），撞声 {out['collisions'] or '无'}")
    if a.cmd in ("speak", "run"):
        from voicecast.speak import speak
        speak(a.script, a.project, n_best=a.n_best, seed=a.seed, engine=eng, cer=not a.no_cer, no_emotion=a.no_emotion)
        print("配音完成：", a.project)


if __name__ == "__main__":
    main()
