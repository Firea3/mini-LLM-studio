"""数据层：统一读取 ``data/processed`` 下的训练/评估数据，并提供划分工具。

数据文件由 ``scripts/build_data.py`` 从 ``data/raw`` 生成，格式见团队契约。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypeVar

from minillm.utils.io import read_jsonl

T = TypeVar("T")


@dataclass(frozen=True)
class SFTExample:
    """一条指令微调样本：prompt -> response。"""

    prompt: str
    response: str
    doc_id: str = ""


@dataclass(frozen=True)
class PreferenceExample:
    """一条偏好样本：同一 prompt 下 chosen 优于 rejected。"""

    prompt: str
    chosen: str
    rejected: str
    doc_id: str = ""


def load_corpus(path: str | Path) -> str:
    """读取预训练纯文本语料。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"语料不存在: {p}（先运行 python scripts/build_data.py）")
    return p.read_text(encoding="utf-8")


def load_sft_examples(path: str | Path) -> list[SFTExample]:
    rows = read_jsonl(path)
    return [
        SFTExample(prompt=str(r["prompt"]), response=str(r["response"]), doc_id=str(r.get("doc_id", "")))
        for r in rows
    ]


def load_preference_examples(path: str | Path) -> list[PreferenceExample]:
    rows = read_jsonl(path)
    return [
        PreferenceExample(
            prompt=str(r["prompt"]),
            chosen=str(r["chosen"]),
            rejected=str(r["rejected"]),
            doc_id=str(r.get("doc_id", "")),
        )
        for r in rows
    ]


def load_eval_qa(path: str | Path) -> list[dict[str, Any]]:
    """RAG 评估集：query / gold_doc_ids / gold_keywords。"""
    return read_jsonl(path)


def load_agent_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Agent 评估集：task / expected_keywords / expected_tools。"""
    return read_jsonl(path)


def train_val_split(items: Sequence[T], val_ratio: float = 0.1, seed: int = 0) -> tuple[list[T], list[T]]:
    """按比例随机划分训练/验证集（固定种子，结果可复现）。"""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 [0, 1) 内")
    data = list(items)
    rng = random.Random(seed)
    rng.shuffle(data)
    n_val = int(len(data) * val_ratio)
    return data[n_val:], data[:n_val]
