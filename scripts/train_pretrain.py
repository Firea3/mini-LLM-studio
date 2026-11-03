"""预训练脚本：字符级语言建模（产出 artifacts/tokenizer.json 与 artifacts/pretrain/model.pt）。

用法::

    python scripts/train_pretrain.py --config configs/quick.yaml
    python scripts/train_pretrain.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 支持 `python scripts/train_pretrain.py` 直接运行
    sys.path.insert(0, str(ROOT))

from minillm.config import load_config  # noqa: E402
from minillm.train.pretrain import run_pretrain  # noqa: E402

DEFAULT_CONFIG = str(ROOT / "configs" / "default.yaml")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="MiniLLM Studio：TinyGPT 字符级预训练")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件路径")
    parser.add_argument("--processed-dir", default=None, help="数据目录（默认 data/processed）")
    parser.add_argument("--artifacts-dir", default=None, help="产物目录（默认 artifacts）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口：加载配置 -> 训练 -> 打印指标摘要。"""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    metrics = run_pretrain(cfg, processed_dir=args.processed_dir, artifacts_dir=args.artifacts_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
