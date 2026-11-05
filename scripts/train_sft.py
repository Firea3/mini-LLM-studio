"""SFT 脚本：在 sft.jsonl 上做指令微调（产出 artifacts/sft/model.pt）。

用法::

    python scripts/train_sft.py --config configs/quick.yaml
    python scripts/train_sft.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 支持 `python scripts/train_sft.py` 直接运行
    sys.path.insert(0, str(ROOT))

from minillm.config import load_config  # noqa: E402
from minillm.train.sft import run_sft  # noqa: E402

DEFAULT_CONFIG = str(ROOT / "configs" / "default.yaml")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="MiniLLM Studio：TinyGPT 监督微调（SFT）")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件路径")
    parser.add_argument("--processed-dir", default=None, help="数据目录（默认 data/processed）")
    parser.add_argument("--artifacts-dir", default=None, help="产物目录（默认 artifacts）")
    parser.add_argument("--init-ckpt", default=None, help="初始检查点（默认 artifacts/pretrain/model.pt）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口：加载配置 -> 训练 -> 打印指标摘要。"""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    metrics = run_sft(
        cfg,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        init_ckpt=args.init_ckpt,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
