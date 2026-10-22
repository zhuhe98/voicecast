"""声音卡 v2：Agent 按人设填写的结构化声音描述（voicecast/SKILL.md）→ 校验 → 渲染为提示词。

卡片字段：age、pitch、quality（1–3 个词）、resonance（可省）、pace、energy、intonation、attitude、ref_line、rationale。
取值范围由类型库（types.yaml：age、pitch_range、vocab）约束；校验不通过的卡片不生成。
"""
from voicecast.library import LIBRARY

FIELDS = ["age", "pitch", "quality", "pace", "energy", "intonation", "attitude", "ref_line"]
REF_BANNED = ["！", "!", "～", "~"]   # 参考台词须为陈述句：感叹、撒娇拖音会抬高音高、让声音变幼（v3 阿瑶、米娅、苏小宇）


def check_ref_line(line):
    line = str(line or "").strip()
    probs = [] if 8 <= len(line) <= 40 else [f"ref_line 长度 {len(line)}，应为 8–40 字"]
    return probs + [f"ref_line 含 {c!r}：参考台词用陈述句，不用感叹或撒娇语气" for c in REF_BANNED if c in line]
ORD = {1: "first", 2: "second"}


def age_phrase(noun, age, gender):
    if age < 20:
        return f"{noun} around {age} years old"
    pron = "her" if gender == "F" else "his"
    part = "early" if age % 10 <= 3 else "mid" if age % 10 <= 6 else "late"
    decade = {20: "twenties", 30: "thirties", 40: "forties", 50: "fifties", 60: "sixties", 70: "seventies",
              80: "eighties", 90: "nineties"}[age // 10 * 10]
    return f"{noun} in {pron} {part} {decade}"


def validate(vtype, card):
    """返回问题列表（空 = 通过）。"""
    lib = LIBRARY
    if vtype not in lib["types"]:
        return [f"未知类型 {vtype}"]
    t, v = lib["types"][vtype], lib["vocab"]
    probs = [f"缺少字段 {f}" for f in FIELDS if f not in card]
    if probs:
        return probs
    if not (t["age"][0] <= card["age"] <= t["age"][1]):
        probs.append(f"age {card['age']} 超出类型范围 {t['age']}")
    scale = lib["pitch_scale"]
    if card["pitch"] not in scale:
        probs.append(f"pitch {card['pitch']!r} 不在 {scale}")
    elif not (t["pitch_range"][0] <= scale.index(card["pitch"]) <= t["pitch_range"][1]):
        probs.append(f"pitch {card['pitch']} 超出类型允许的 {scale[t['pitch_range'][0]]}–{scale[t['pitch_range'][1]]}")
    q = card["quality"] if isinstance(card["quality"], list) else [card["quality"]]
    if not 1 <= len(q) <= 3:
        probs.append("quality 应为 1–3 个词")
    probs += [f"quality {x!r} 不在词表" for x in q if x not in v["quality"]]
    if card.get("resonance") and card["resonance"] not in v["resonance"]:
        probs.append(f"resonance {card['resonance']!r} 不在 {v['resonance']}")
    for f in ("pace", "energy", "intonation"):
        if card[f] not in v[f]:
            probs.append(f"{f} {card[f]!r} 不在 {v[f]}")
    att = str(card["attitude"]).lower()
    if len(att.split()) > lib["attitude_max_words"]:
        probs.append(f"attitude 超过 {lib['attitude_max_words']} 个词")
    probs += [f"attitude 含禁用词 {w!r}（不写角色扮演，只写可听的语气）" for w in lib["attitude_banned"] if w in att]
    return probs + check_ref_line(card["ref_line"])


def render(vtype, card):
    t = LIBRARY["types"][vtype]
    q = card["quality"] if isinstance(card["quality"], list) else [card["quality"]]
    qs = q[0] if len(q) == 1 else ", ".join(q[:-1]) + " and " + q[-1]
    res = f" with {card['resonance']} resonance" if card.get("resonance") else ""
    head = age_phrase(t["noun"], card["age"], t["gender"])
    return (f"{head[0].upper()}{head[1:]}, {card['pitch']} pitch, {qs} timbre{res}; "
            f"{card['pace']} pace, {card['energy']} energy, {card['intonation']} intonation, {card['attitude']} tone.")


# ── 声音卡 v3：身份由类型锚定，Agent 只在偏离类型默认处做少量修改（v1/v2 人耳对比后的结论）──
V3_PACE = ["slow", "medium", "fast"]   # 不允许极端值（very slow / very fast 会串进年龄感）


def validate_v3(vtype, card):
    lib = LIBRARY
    if vtype not in lib["types"]:
        return [f"未知类型 {vtype}"]
    probs = [f"缺少字段 {f}" for f in ("attitude", "ref_line") if not card.get(f)]
    if card.get("pitch", "lo") not in ("lo", "hi"):
        probs.append("pitch 只能是 lo / hi（类型内两档）")
    q = card.get("quality") or []
    q = q if isinstance(q, list) else [q]
    if len(q) > 2:
        probs.append("quality 最多 2 个词")
    probs += [f"quality {x!r} 不在词表" for x in q if x not in lib["vocab"]["quality"]]
    if card.get("pace") and card["pace"] not in V3_PACE:
        probs.append(f"pace 只能是 {V3_PACE}（不用极端值）")
    att = str(card.get("attitude", "")).lower()
    if len(att.split()) > lib["attitude_max_words"]:
        probs.append(f"attitude 超过 {lib['attitude_max_words']} 个词")
    probs += [f"attitude 含禁用词 {w!r}" for w in lib["attitude_banned"] if w in att]
    return probs + (check_ref_line(card["ref_line"]) if card.get("ref_line") else [])


def render_v3(vtype, card):
    """类型身份 + 类型音高档 [+ 音质] ; 类型演绎（语速可改，语气换成 Agent 的 attitude）。"""
    t = LIBRARY["types"][vtype]
    q = card.get("quality") or []
    q = q if isinstance(q, list) else [q]
    qs = f", {q[0] if len(q) == 1 else ' and '.join(q)} timbre" if q else ""
    parts = [x.strip() for x in t["delivery"].split(",")]
    parts = [f"{card['pace']} pace" if (card.get("pace") and x.endswith("pace")) else x for x in parts]
    if t.get("keep_tone"):   # 老年：保留类型语气中的年龄词，attitude 追加
        parts = parts + [f"{card['attitude']} tone"]
    else:
        parts = [x for x in parts if not x.endswith("tone")] + [f"{card['attitude']} tone"]
    return f"{t['identity_base']}, {t['pitch'][card.get('pitch', 'lo')]} pitch{qs}; {', '.join(parts)}."


def check(vtype, card):
    """按卡片版本校验：version 3 → validate_v3，否则 v2。"""
    return validate_v3(vtype, card) if card.get("version") == 3 else validate(vtype, card)


def to_prompt(vtype, card):
    return render_v3(vtype, card) if card.get("version") == 3 else render(vtype, card)
