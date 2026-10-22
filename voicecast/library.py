"""类型库（types.yaml）与按类型默认生成的提示词。只依赖 PyYAML，不加载任何模型。"""
from pathlib import Path

import yaml

LIBRARY = yaml.safe_load((Path(__file__).parent / "types.yaml").read_text(encoding="utf-8"))


def type_prompt(vtype, level="lo"):
    """没有声音卡时的类型默认提示词：身份 + 音高档 + 默认说话方式。"""
    t = LIBRARY["types"][vtype]
    return f"{t['identity_base']}, {t['pitch'][level]} pitch; {t['delivery']}."
