---
name: voicecast
description: 给一部作品的一整组角色选声音并配音：读剧本 / 小说 / 角色设定 → 判类型、写声音卡 → 先问用户要哪种输出：A 每个角色的声音描述（给其他模型用，不需要显卡），或 B 本地整组挑选出彼此分得清的原创声音并导出参考录音（可拷贝），可继续逐句配音。用于 AI 短剧、漫剧、有声书、游戏的多角色配音。
---

# voicecast：群像声音选角

分工：
- **你（Agent）**负责语言层面的判断：读材料、列角色、判类型、把人设写成声音卡、整理台词表、根据用户的试听意见改卡。
- **本地工具**负责声音：生成候选、检查年龄性别、整组挑选（让角色彼此不像）、锁定、逐句配音。你写的卡片会被校验，不合格会报错退回。

所有命令都在 voicecast 仓库根目录运行。

## 第 0 步：先问用户要哪种输出

开始前用一句话问用户，让用户自己选（可以两种都要）：

> 你要哪种结果？
> **A. 声音描述**：每个角色一段文字描述（英文提示词 + 试音台词 + 合理音高范围），拿去给别的语音模型用，或另作他用。不需要显卡，几秒钟出结果；但只靠文字，不能保证角色之间一定分得清。
> **B. 声音本身**：在本机生成并挑选，输出每个角色的参考录音（可直接拷贝，用于任何支持参考音频克隆的语音模型），并保证整组角色彼此分得清。需要 NVIDIA 显卡，选角约几分钟。

用户已经说清楚要哪种时不用再问。用户没有显卡时只能选 A。

- 选 **A**：做第 1–5 步，然后运行 `python -m voicecast prompts characters.yaml --out <输出目录>`。把 `prompts.md` 里的表格和"可能撞声的组合"提醒展示给用户，告诉用户文件位置（`prompts.md` 给人看，`prompts.yaml` 给程序用）。用户需要中文描述时，由你把英文提示词翻译给用户看，文件里保留英文原文。
- 选 **B**：做第 1–8 步。锁定后声音会自动导出到 `projects/<名字>/export/`（另有同名 zip），把这个位置告诉用户。
- 两种都要：先做 A，再接着做 B 的第 6 步起。

## 流程

1. **列角色**：从剧本 / 小说 / 设定中列出所有说话的角色（含旁白），每人一行简短描述（年龄、身份、性格、说话习惯）。
2. **判类型**：读 `voicecast/types.yaml`，为每个角色选一个最贴近的类型（看 `zh` 中文名、`hint` 判断说明、`age` 年龄范围、`gender`）。类型决定性别、年龄感和音高范围，选错类型会直接选错声音。
3. **写声音卡**（见下）。
4. **整组检查**：同一阵容里同类型或相邻类型的角色（如元气少女和萝莉、两个冷峻青年），在 `pitch`（一高一低）、`attitude`、`ref_line` 风格上要明显不同；不要为了拉开而改身份。
5. **写角色表** `characters.yaml`：`characters: [{id, name, type, description, intro, card}]`。`intro` 是一句自我介绍，用于阵容试音。（A 到此为止，接着运行 `prompts` 命令。）
6. **选声音**（B），两种方式任选：
   - 让用户自己挑（推荐）：`python -m voicecast.studio characters.yaml --project projects/<名字>`，然后告诉用户在浏览器打开 `http://127.0.0.1:7873`。每个角色给 3 个已经避开其他角色的选项，用户可以"换一批"，选好后点锁定。
   - 全自动：`python -m voicecast lineup characters.yaml --project projects/<名字>`，选角并生成阵容试音（`projects/<名字>/lineup/episode.wav`）。
7. **按用户意见修改**：用户说"店小二再热情一点""这两个太像"时，只改对应角色的卡（或类型），重跑第 6 步。
   锁定后的声音在 `projects/<名字>/export/`：每个角色一个文件夹（`reference.wav` 参考录音、`reference.txt` 对应文字、`sample.wav` 试听），加上 `voices.yaml` 清单和 `export.zip`。需要重新导出时运行 `python -m voicecast export --project projects/<名字>`。
8. **配音**：把台词整理成台词表（格式见下），运行 `python -m voicecast speak script.yaml --project projects/<名字>`。输出整段音频、SRT 字幕和质量报告。

## 声音卡（card，v3）

**身份（性别、年龄感、音高区间）由类型固定，你只在角色明显偏离类型默认的地方做少量修改。**
实验结论：让 Agent 从零重写整段声音描述，说话方式的词会串进身份——高能量 + 拖腔 + 明亮会变幼、变女；低沉 + 慢 + 低能量会变老；人设接近类型默认的角色被改写后反而变差。

| 字段 | 取值 | 怎么填 |
|---|---|---|
| `version` | `3` | 固定 |
| `pitch` | `lo` / `hi`（类型内两档），默认 `lo` | 只有人设明确偏高（更年轻、更尖、更活泼）才用 `hi`；同类型两个角色可一高一低 |
| `quality` | 0–2 个词，来自 `types.yaml` 的 `vocab.quality`，可省 | 只在人设明确写到嗓音特点时填（沙哑、苍老、清冷、甜）；没写就省略 |
| `pace` | `slow` / `medium` / `fast`，可省 | 只在人设明确偏离类型默认语速时填（伙计吆喝 → fast） |
| `attitude` | ≤ 6 个英文词 | **必填**，角色说话的语气，如 `eager, fawning and cheerful`；不写 like / as if / character 等角色扮演措辞 |
| `ref_line` | 一句中文**陈述句**，8–40 字 | **必填**，用词体现身份与年龄（"孩子""客官""本座"），**语气平稳**：不用感叹号、不撒娇拖音、不带强烈情绪。这句用来生成并锁定声音，感叹和撒娇会抬高音高、让声音变幼；情绪留给逐句配音 |
| `rationale` | 一句中文 | 依据，给用户看 |

原则：
- **少改**：人设与类型默认一致的角色，只写 `attitude` 和 `ref_line`。
- **语气和参考台词是区分角色最有效、也最安全的手段**；音高档、音质、语速只在人设明确需要时动。
- 只描述听得到的声音，不写外貌、剧情、比喻；情绪（哭、怒）不写进卡片。
- 工具会按类型的音高范围检查候选；全部不合格时会自动把音高降为低档、去掉语速修改再试，并在报告里标出。

## 角色表示例

```yaml
characters:
  - id: waiter
    name: 店小二
    type: sunny_youth
    description: 客栈伙计，机灵热情
    intro: 客官里边请！小店的酒是全城最香的，您要几斤？
    card:
      version: 3
      pace: fast
      attitude: eager, fawning and cheerful
      ref_line: 客官里边请，楼上雅座，好酒好菜马上就到。
      rationale: 伙计招揽客人：语速快、殷勤热情；身份与音高沿用类型
  - id: su
    name: 苏念
    type: gentle_heroine
    description: 二十二岁，倔强善良的女主
    intro: 我叫苏念。日子再难，我也会靠自己一步一步走下去。
    card:
      version: 3
      attitude: gentle yet quietly stubborn
      ref_line: 我不需要你的同情，我自己能行。
      rationale: 人设与类型默认一致，只写语气与台词
```

## 台词表

```yaml
scenes:
  - id: s1
    title: 客栈
    lines:
      - {speaker: waiter, text: 客官里边请，楼上雅座。}
      - {speaker: su, emotion: sad, text: 我只是想找个地方歇一歇。}
```

`speaker` 是角色表里的 `id`。`emotion` 可省（平静），可选 `happy`、`angry`、`sad`、`surprised`。平静的句子直接克隆锁定的声音；带情绪的句子用锁定的声音加情绪描述生成，每句生成 3 个候选，自动挑情绪最对、声音最稳的一个。
