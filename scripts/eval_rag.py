"""RAG 检索评估脚本：``python scripts/eval_rag.py [--config configs/default.yaml]``。

默认评估两组数据（都写入 ``reports/app_metrics.json``）：

- ``rag``      ：``data/processed/eval_qa.jsonl``（FAQ 问法改写，词法重合高，指标偏乐观）；
- ``rag_hard`` ：内置的语义改写难例（``HARD_EVAL_QUERIES``，人工构造，用来暴露纯词法
  BM25 在语义泛化上的不足）。

指标：recall@1 / recall@3 / MRR / 平均检索延迟。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接 ``python scripts/eval_rag.py`` 运行（把仓库根目录加入 sys.path）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.config import load_config  # noqa: E402
from minillm.evaluation.rag_eval import HARD_EVAL_QUERIES, evaluate_rag  # noqa: E402
from minillm.rag import build_rag_pipeline  # noqa: E402
from minillm.utils.metrics import save_metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估 MiniLLM Studio 的 RAG 检索质量")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="配置文件路径")
    parser.add_argument("--eval-path", default=None, help="eval_qa.jsonl 路径（默认 data/processed/eval_qa.jsonl）")
    parser.add_argument("--top-k", type=int, default=3, help="评估时检索条数（recall@3 至少取 3）")
    parser.add_argument("--no-hard", action="store_true", help="跳过内置语义改写难例评估")
    parser.add_argument("--no-save", action="store_true", help="只打印指标，不写入 reports/app_metrics.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    eval_path = Path(args.eval_path) if args.eval_path else cfg.processed_dir() / "eval_qa.jsonl"
    if not eval_path.exists():
        print(f"[eval_rag] 评估集不存在: {eval_path}（先运行 python -m minillm data）", file=sys.stderr)
        return 1
    pipeline = build_rag_pipeline(cfg)
    reports_path = cfg.path("reports_dir") / "app_metrics.json"

    results: dict[str, dict] = {}
    results["rag"] = evaluate_rag(pipeline, eval_path, top_k=args.top_k)
    if not args.no_save:
        save_metrics(reports_path, "rag", results["rag"])
    if not args.no_hard:
        hard_metrics = evaluate_rag(pipeline, HARD_EVAL_QUERIES, top_k=args.top_k)
        results["rag_hard"] = {
            **hard_metrics,
            "note": "人工构造的语义改写难例（非训练数据；gold 来自 eval 划分文档 kb-campus-07/10）",
        }
        if not args.no_save:
            save_metrics(reports_path, "rag_hard", results["rag_hard"])
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
