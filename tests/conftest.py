"""pytest 公共配置：把仓库根目录加入 sys.path，并暴露通用 fixture。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from minillm.config import default_config  # noqa: E402


@pytest.fixture()
def repo_root() -> Path:
    return ROOT


@pytest.fixture()
def tiny_cfg():
    """极小的模型/训练配置，单元测试用（1~3 步内完成）。"""
    cfg = default_config()
    cfg.model.vocab_size = 64
    cfg.model.block_size = 32
    cfg.model.n_layer = 1
    cfg.model.n_head = 2
    cfg.model.n_embd = 32
    cfg.model.dropout = 0.0
    cfg.model.init_std = 0.02
    cfg.pretrain.max_steps = 2
    cfg.pretrain.batch_size = 2
    cfg.pretrain.warmup_steps = 1
    cfg.pretrain.eval_interval = 1
    cfg.pretrain.eval_batches = 1
    cfg.pretrain.log_interval = 1
    cfg.sft.epochs = 1
    cfg.sft.batch_size = 2
    cfg.sft.log_interval = 1
    cfg.reward.epochs = 1
    cfg.reward.batch_size = 2
    cfg.reward.log_interval = 1
    cfg.ppo.max_steps = 1
    cfg.ppo.batch_prompts = 2
    cfg.ppo.horizon = 8
    cfg.ppo.max_new_tokens = 8
    cfg.ppo.ppo_epochs = 1
    cfg.ppo.minibatch_size = 2
    cfg.ppo.log_interval = 1
    cfg.dqn.episodes = 2
    cfg.dqn.learning_starts = 1
    cfg.dqn.eval_interval = 1
    cfg.rag.top_k = 2
    return cfg
