"""配置层：dataclass 描述全部超参数，支持 YAML 局部覆盖。

设计考量
--------
1. **配置集中**：所有超参集中在 ``ProjectConfig`` 及各子配置中，训练脚本不允许出现魔法数字，
   实验可复现（配置文件 + 随机种子）。
2. **局部覆盖**：YAML 只需要写想改的字段，其余取 dataclass 默认值。
3. **严格校验**：未知字段直接报错，避免拼写错误被静默忽略；加载后做基本合法性检查。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 仓库根目录：minillm/config.py -> parents[1]
REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """配置非法或 YAML 中出现了未知字段。"""


@dataclass
class PathsConfig:
    """目录约定（相对路径以仓库根目录解析）。"""

    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"


@dataclass
class ModelConfig:
    """TinyGPT 结构超参。"""

    vocab_size: int = 1200
    block_size: int = 128
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 96
    dropout: float = 0.1
    tie_weights: bool = True
    init_std: float = 0.02


@dataclass
class PretrainConfig:
    """预训练（自回归语言建模）配置。"""

    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 0.01
    max_steps: int = 300
    warmup_steps: int = 20
    eval_interval: int = 50
    eval_batches: int = 5
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    seed: int = 42
    device: str = "cpu"
    log_interval: int = 50
    min_freq: int = 1
    max_vocab: int = 3000


@dataclass
class SFTConfig:
    """监督微调配置（可选 LoRA）。"""

    epochs: int = 6
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    seed: int = 0
    device: str = "cpu"
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    log_interval: int = 20


@dataclass
class RewardConfig:
    """奖励模型（Bradley-Terry 偏好建模）配置。"""

    epochs: int = 6
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_ratio: float = 0.2
    seed: int = 0
    device: str = "cpu"
    margin: float = 0.0
    log_interval: int = 20


@dataclass
class PPOConfig:
    """RLHF 的 PPO 配置（token 级 KL 惩罚 + GAE）。"""

    max_steps: int = 80
    batch_prompts: int = 8
    horizon: int = 48
    lr: float = 2e-5
    value_lr: float = 5e-4
    clip: float = 0.2
    kl_coef: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    ppo_epochs: int = 2
    minibatch_size: int = 8
    entropy_coef: float = 0.01
    temperature: float = 0.8
    top_k: int = 0
    seed: int = 0
    device: str = "cpu"
    log_interval: int = 5
    max_new_tokens: int = 48


@dataclass
class DQNConfig:
    """经典强化学习（GridWorld + DQN）配置。"""

    episodes: int = 150
    max_steps: int = 50
    gamma: float = 0.95
    lr: float = 1e-3
    buffer_size: int = 5000
    batch_size: int = 32
    learning_starts: int = 100
    target_update: int = 20
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.985
    eval_interval: int = 25
    eval_episodes: int = 5
    hidden: int = 64
    seed: int = 0
    device: str = "cpu"
    log_interval: int = 25


@dataclass
class RAGConfig:
    """检索增强生成配置。"""

    top_k: int = 3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    min_score: float = 0.0
    max_context_chars: int = 800
    snippet_chars: int = 120
    use_native: bool = False
    guard_model_output: bool = True


@dataclass
class AgentConfig:
    """ReAct Agent 配置。"""

    max_steps: int = 5
    planner: str = "rule"
    tools: list[str] = field(
        default_factory=lambda: ["calculator", "knowledge_search", "text_stats", "unit_convert"]
    )


@dataclass
class ServeConfig:
    """HTTP 服务与模型加载配置。"""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    policy_ckpt: str = "artifacts/ppo/model.pt"
    fallback_policy_ckpt: str = "artifacts/sft/model.pt"
    tokenizer_path: str = "artifacts/tokenizer.json"
    reward_ckpt: str = "artifacts/reward/model.pt"
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 0
    rag_top_k: int = 3
    allow_fallback: bool = True


@dataclass
class ProjectConfig:
    """项目级配置（一个 YAML 文件对应一棵配置树）。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # ------------------------------------------------------------------ 路径
    def path(self, name: str) -> Path:
        """把 paths 配置项解析为绝对路径（相对仓库根目录）。"""
        try:
            raw = getattr(self.paths, name)
        except AttributeError as exc:  # 帮助排查配置拼写错误
            raise ConfigError(f"未知路径配置项: paths.{name}") from exc
        p = Path(raw)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def processed_dir(self) -> Path:
        return self.path("data_dir") / "processed"

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """基本合法性检查，尽早暴露配置错误。"""
        if self.model.n_embd % self.model.n_head != 0:
            raise ConfigError("model.n_embd 必须能被 model.n_head 整除")
        if self.model.block_size < 16:
            raise ConfigError("model.block_size 过小")
        for name in ("pretrain", "sft", "reward", "ppo", "dqn"):
            section = getattr(self, name)
            if getattr(section, "device") not in ("cpu", "cuda"):
                raise ConfigError(f"{name}.device 只能是 cpu/cuda")
        if self.ppo.clip <= 0 or self.ppo.kl_coef < 0:
            raise ConfigError("ppo.clip 必须为正、ppo.kl_coef 不能为负")
        if self.rag.top_k <= 0:
            raise ConfigError("rag.top_k 必须为正整数")
        if self.agent.planner not in ("rule", "llm"):
            raise ConfigError("agent.planner 只能是 rule 或 llm")

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# YAML 的顶层分节 -> dataclass 类型
_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "pretrain": PretrainConfig,
    "sft": SFTConfig,
    "reward": RewardConfig,
    "ppo": PPOConfig,
    "dqn": DQNConfig,
    "rag": RAGConfig,
    "agent": AgentConfig,
    "serve": ServeConfig,
    "paths": PathsConfig,
}


def _coerce(current: Any, value: Any, where: str) -> Any:
    """按默认值的类型做轻量类型转换，避免 YAML 里的 "8" 变成字符串。"""
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} 需要 bool，实际为 {type(value).__name__}")
        return value
    if isinstance(current, int):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where} 需要 int，实际为 {value!r}") from exc
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where} 需要 float，实际为 {value!r}") from exc
    if isinstance(current, list) and isinstance(value, list):
        return list(value)
    return value


def _update_section(section_obj: Any, data: dict[str, Any], section_name: str) -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"配置节 {section_name} 必须是映射（dict）")
    for key, value in data.items():
        if not hasattr(section_obj, key):
            raise ConfigError(f"未知配置项: {section_name}.{key}")
        current = getattr(section_obj, key)
        where = f"{section_name}.{key}"
        if hasattr(current, "__dataclass_fields__"):
            _update_section(current, value, where)
        else:
            setattr(section_obj, key, _coerce(current, value, where))


def default_config() -> ProjectConfig:
    """返回默认配置（与 configs/default.yaml 等价）。"""
    cfg = ProjectConfig()
    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """从 YAML 加载配置；``None`` 表示默认配置。

    YAML 采用"局部覆盖"语义：只写需要改的字段，其余取默认值。
    """
    cfg = default_config()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("配置文件顶层必须是映射（dict）")
    for name, section_data in raw.items():
        if name not in _SECTIONS:
            raise ConfigError(f"未知配置节: {name}（可选: {', '.join(_SECTIONS)}）")
        _update_section(getattr(cfg, name), section_data, name)
    cfg.validate()
    return cfg


def save_config(cfg: ProjectConfig, path: str | Path) -> None:
    """把生效配置落盘，便于实验复现。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
