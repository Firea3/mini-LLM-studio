"""PPO 脚本：RLHF 对齐（需要 artifacts/sft/model.pt 与 artifacts/reward/model.pt）。

用法::

    python scripts/train_ppo.py --config configs/quick.yaml
    python scripts/train_ppo.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 支持 `python scripts/train_ppo.py` 直接运行
    sys.path.insert(0, str(ROOT))

from minillm.config import load_config  # noqa: E402
from minillm.train.ppo import run_ppo  # noqa: E402

DEFAULT_CONFIG = str(ROOT / "configs" / "default.yaml")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="MiniLLM Studio：PPO(RLHF) 对齐训练")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件路径")
    parser.add_argument("--processed-dir", default=None, help="数据目录（默认 data/processed）")
    parser.add_argument("--artifacts-dir", default=None, help="产物目录（默认 artifacts）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口：加载配置 -> 训练 -> 打印指标摘要（样本截断，避免刷屏）。"""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    metrics = run_ppo(cfg, processed_dir=args.processed_dir, artifacts_dir=args.artifacts_dir)
    summary = dict(metrics["ppo"])
    summary.pop("samples", None)  # 样本已写入 reports/ppo_samples.json
    print(json.dumps({"ppo": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
