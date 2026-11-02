"""Agent 任务评估脚本：``python scripts/eval_agent.py [--config configs/default.yaml]``。

指标：任务成功率 / 平均步数 / 平均延迟；结果合并写入
``reports/app_metrics.json`` 的 ``agent`` section。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接 ``python scripts/eval_agent.py`` 运行（把仓库根目录加入 sys.path）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.agent import ReActAgent, build_planner, build_tools  # noqa: E402
from minillm.config import load_config  # noqa: E402
from minillm.evaluation.agent_eval import evaluate_agent  # noqa: E402
from minillm.rag import build_rag_pipeline  # noqa: E402
from minillm.utils.metrics import save_metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估 MiniLLM Studio 的 ReAct Agent 任务成功率")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="配置文件路径")
    parser.add_argument("--tasks-path", default=None, help="agent_tasks.jsonl 路径（默认 data/processed/agent_tasks.jsonl）")
    parser.add_argument("--no-save", action="store_true", help="只打印指标，不写入 reports/app_metrics.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tasks_path = Path(args.tasks_path) if args.tasks_path else cfg.processed_dir() / "agent_tasks.jsonl"
    if not tasks_path.exists():
        print(f"[eval_agent] 评估集不存在: {tasks_path}（先运行 python -m minillm data）", file=sys.stderr)
        return 1
    pipeline = build_rag_pipeline(cfg)
    agent = ReActAgent(
        build_planner(cfg, pipeline),
        build_tools(cfg, pipeline),
        max_steps=cfg.agent.max_steps,
    )
    metrics = evaluate_agent(agent, tasks_path)
    if not args.no_save:
        save_metrics(cfg.path("reports_dir") / "app_metrics.json", "agent", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
