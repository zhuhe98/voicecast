"""模型下载与加载：固定 Hugging Face revision，首次使用时下载到本地缓存（HF_HOME）。"""
import os

DEVICE = os.environ.get("VOICECAST_DEVICE", "cuda:0")
# 精度与批大小：命令行 --dtype / --batch 优先，其次环境变量。显存小的显卡用 bf16 + 小批量
DTYPE = os.environ.get("VOICECAST_DTYPE", "fp32")
BATCH = int(os.environ.get("VOICECAST_BATCH", "16"))
REVISIONS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "fd4b254389122332181a7c3db7f27e918eec64e3",
    "funasr/campplus": "e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
    "emotion2vec/emotion2vec_plus_large": "6c303ba987b86b93193de93e34bb2b077a6bedc4",
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
}
# 只下载需要的文件（whisper 仓库含多种格式，全量约 24 GB）
ALLOW = {"openai/whisper-large-v3": ["*.json", "*.txt", "model.safetensors"]}


def add_engine_args(ap):
    """给命令行加上 --dtype / --batch（默认取环境变量 VOICECAST_DTYPE / VOICECAST_BATCH）。"""
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default=DTYPE,
                    help="模型精度（默认 fp32；30/40 系等支持 bf16 的显卡建议 bf16，显存约减半）")
    ap.add_argument("--batch", type=int, default=BATCH, help="每批同时生成几句（默认 16；显存不足时调小）")


def model_path(repo):
    """固定 revision 的本地快照路径（缓存中没有时下载）。"""
    from huggingface_hub import snapshot_download   # 延迟导入：prompts 命令只需 PyYAML
    return snapshot_download(repo, revision=REVISIONS[repo], allow_patterns=ALLOW.get(repo))
