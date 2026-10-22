# voicecast · 群像声音选角

给一部作品里的一整组角色挑声音，让每个角色的声音都分得清、合身份、选定后不变。适用于 AI 短剧、漫剧、有声书、游戏的多角色配音。

**演示页**：<https://zhuhe98.github.io/voicecast/>（听四组阵容、看整组挑选前后的对比）

所有声音都是模型按文字描述合成的原创声音，不克隆任何真人。

## 解决什么问题

一次要配六个、十个角色时，每个角色单独生成一个合适的声音不难，难的是放在一起时观众仍然听得出谁是谁。

- **撞声**：每个角色各自挑一个合适的声音，没人管它们彼此像不像。演示中的校园剧，元气少女和萝莉两个声音的相似度达到 0.83（0 到 1，越接近 1 越像同一个人）；整组一起挑之后降到 0.67。
- **年龄性别跑偏**：只靠文字描述，"十七岁男孩，活泼爱撒娇"会被生成成女声。voicecast 让年龄、性别、音高由类型固定，Agent 只写语气。
- **声音不稳定**：每次生成都是一个新的人。voicecast 选定后锁定声音，之后每一句都克隆这个声音。

## 怎么做的

1. **定类型**：Agent 读角色表，从 22 个类型（霸道总裁、元气少女、老年男性、旁白……）中为每个角色选一个，并写一张声音卡（语气 + 一句参考台词）。
2. **备候选**：每个角色从音色库取一批同类型的声音，再按声音卡现场生成几个；检查音高是否在类型的合理范围内（拦住性别、年龄跑偏）。
3. **整组挑选**：先让每个角色各拿一个候选，再反复找出全组最像的两个角色，给其中一个换候选，直到最像的一对无法再降低。
4. **人来拍板**：每个角色给 3 个已经避开其他角色的选项，用户选一个，不满意可以换一批。（实验表明 AI 判断不了"这个声音像不像这个角色"，所以最后一步交给人。）
5. **锁定并配音**：选定的声音存为参考录音；平静的台词直接克隆，带情绪的台词用锁定的声音加情绪描述生成，每句生成 3 个候选自动挑最好的。

方法细节和实验结论见 [METHOD.md](METHOD.md)。

## 两种输出

| | A. 声音描述 | B. 声音本身 |
|---|---|---|
| 得到什么 | 每个角色一段英文声音描述 + 试音台词 + 合理音高范围（`prompts.md` / `prompts.yaml`） | 每个角色的参考录音 + 对应文字 + 试听样本（`export/` 文件夹与 zip） |
| 用途 | 交给其他语音模型生成，或另作他用 | 拷贝到任何支持参考音频克隆的语音模型；或直接用 voicecast 逐句配音 |
| 角色之间分得清 | 只做文字层面的提醒（同类型且音高、语速都相同时警告） | 实际生成并测量，整组挑选保证 |
| 需要 | 只需 PyYAML，不需要显卡 | NVIDIA 显卡 |

作为 Agent 技能使用时，Agent 会先问你要哪一种。

## 两种用法

### 作为 Agent 技能（Claude Code、Codex 等）

`skills/voicecast/SKILL.md` 是给 Agent 的工作规范：怎么列角色、判类型、写声音卡、调用本地工具、按用户意见修改。

Claude Code：把 `skills/voicecast` 文件夹复制到 `~/.claude/skills/`（或项目里的 `.claude/skills/`），然后直接说"用 voicecast 给这段小说的角色配音"，并给出 voicecast 仓库的位置。其他 Agent：把 SKILL.md 的内容放进它的系统提示或项目说明。

Agent 只负责语言判断（读材料、写卡片），声音的生成、测量和挑选全部由本地工具完成，所以不依赖特定的大模型。

### 直接用命令行

自己按 `examples/` 的格式写角色表，然后运行下面的命令。

## 安装

需要 NVIDIA 显卡。我们在 Quadro RTX 8000（48 GB）上测试，其他显卡未测试。

```bash
git clone https://github.com/zhuhe98/voicecast.git
cd voicecast
conda create -n voicecast python=3.12 -y && conda activate voicecast
pip install -r requirements.txt
```

首次运行会自动从 Hugging Face 下载模型（约 14 GB，版本已固定）：Qwen3-TTS 1.7B（VoiceDesign 与 Base）、CAM++ 说话人向量、emotion2vec 情绪识别、Whisper large-v3 读字检查。

指定显卡：`VOICECAST_DEVICE=cuda:1`（默认 `cuda:0`）。

### 显存与精度

需要显卡的命令（`cast` / `speak` / `run` / `lineup`、`voicecast.studio`、`voicecast.bank build`）都支持两个参数，也可以用环境变量设置：

| 参数 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `--dtype fp32/bf16/fp16` | `VOICECAST_DTYPE` | `fp32` | 模型精度。RTX 30/40 系等支持 bf16 的显卡建议用 `bf16`，显存约减半；20 系及更早的显卡用 `fp32` |
| `--batch N` | `VOICECAST_BATCH` | `16` | 每批同时生成几句。显存不足（CUDA out of memory）时调小 |

| 显存 | 建议设置 |
|---|---|
| 48 GB | 默认设置（`fp32`，`--batch 16`），已测试 |
| 16–24 GB | `--dtype bf16`（估算，未测试） |
| 8–12 GB | `--dtype bf16 --batch 2`，配音时加 `--no-cer` 跳过 Whisper 读字检查（估算，未测试） |

```bash
python -m voicecast.studio examples/campus.yaml --project projects/campus --dtype bf16 --batch 4
```

Windows 用户：本项目主要在 Linux 上测试。

## 快速开始

```bash
# A. 只输出声音描述（不需要显卡，pip install pyyaml 即可）
python -m voicecast prompts examples/campus.yaml --out out/campus

# B. 生成声音：
# 选角工作台：浏览器打开 http://127.0.0.1:7873 ，每个角色 3 个选项，选择 / 换一批 / 锁定
python -m voicecast.studio examples/campus.yaml --project projects/campus

# 全自动选角 + 阵容试音（每个角色说一句自我介绍）
python -m voicecast lineup examples/campus.yaml --project projects/campus

# 按台词表配音：输出整段音频、SRT 字幕、质量报告
python -m voicecast speak examples/campus_script.yaml --project projects/campus

# 试听审阅页
python -m voicecast.review projects/campus

# 重新导出锁定的声音（锁定时已自动导出到 projects/campus/export/）
python -m voicecast export --project projects/campus
```

### 音色库（可选，推荐）

```bash
python -m voicecast.bank build            # 22 个类型 × 每类 24 个声音，生成后检查音高并入库到 bank/
python -m voicecast.bank build --types loli,genki_girl --n 12   # 只补某几类
```

没有音色库时，工作台只用按声音卡现场生成的候选（每个角色 4 个），选择余地小一些。音色库会记录哪些声音已被哪个项目用过，新项目不会再分到同一个声音。

## 角色表与台词表

格式见 [skills/voicecast/SKILL.md](skills/voicecast/SKILL.md)，类型列表见 [voicecast/types.yaml](voicecast/types.yaml)，示例见 [examples/](examples/)。

## 目录

| 路径 | 内容 |
|---|---|
| `voicecast/types.yaml` | 类型库：22 个类型（二次元、真人短剧、旁白），每类的身份描述、音高两档、音高合理范围 |
| `voicecast/cards.py` | 声音卡校验与渲染 |
| `voicecast/export.py` | 两种输出：声音描述（prompts）、导出锁定的声音（export） |
| `voicecast/bank.py` | 音色库：离线批量生成、检查、入库 |
| `voicecast/studio.py` | 选角工作台（Gradio） |
| `voicecast/cast.py` | 全自动选角与整组挑选算法 |
| `voicecast/speak.py` | 逐句配音、字幕、报告 |
| `voicecast/review.py` | 试听审阅页 |
| `voicecast/backend/` | 模型加载与测量（TTS、说话人向量、情绪、读字、声学测量） |
| `skills/voicecast/` | Agent 技能说明 |
| `examples/` | 示例角色表与台词表 |
| `docs/` | 演示页（GitHub Pages） |

## 已知局限

- 只支持中文台词。
- 老年角色克隆后会损失一些沧桑感。
- 情绪强度、句与句之间的音高跳变取决于 TTS 模型本身，不在本项目范围内。
- 同一类型在 TTS 中可用的声音范围有限，同类型角色很多时（例如四姐妹）仍会偏像。

## 许可

代码以 Apache License 2.0 发布。所用模型（Qwen3-TTS、CAM++、emotion2vec、Whisper）遵循各自的许可，请在对应的模型页面确认。请不要用本项目模仿真实人物的声音。

---

## English

**voicecast** casts voices for a whole ensemble of characters (AI short dramas, comics, audiobooks, games) so that every character sounds distinct, fits their type, and stays the same across lines. An agent (Claude Code, Codex, …) reads the material, assigns each character one of 22 voice types and writes a short voice card; local tools then generate candidates (voice bank + on-the-fly VoiceDesign), check pitch against the type, run a group max-min selection on CAM++ speaker similarity so no two characters sound alike, let a human make the final pick, lock each voice by cloning, and dub scripts line by line with emotion. Built on Qwen3-TTS 1.7B. Chinese only for now. See `skills/voicecast/SKILL.md` and `METHOD.md`.
