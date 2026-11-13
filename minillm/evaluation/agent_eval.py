"""离线评估：Agent 任务成功率。

判定标准（见契约 §3.6）
-----------------------
1. 答案包含 ``expected_keywords`` 中任意一个关键词；
2. 且实际执行过的工具集合是 ``expected_tools`` 的超集（校验"用对了工具"）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minillm.utils.io import read_jsonl
from minillm.utils.logging import get_logger

LOGGER = get_logger(__name__)


def evaluate_agent(agent: Any, tasks_path: str | Path) -> dict[str, Any]:
    """评估 ReAct Agent 的任务成功率。

    Parameters
    ----------
    agent:
        需提供 ``run(task) -> AgentTrace`` 的对象（一般为 ``ReActAgent``）。
    tasks_path:
        评估集路径，JSONL 每行 ``{"task", "expected_keywords", "expected_tools"}``。

    Returns
    -------
    dict
        ``{"num_tasks", "success_rate", "avg_steps", "avg_latency_ms", "details"}``。
    """
    path = Path(tasks_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Agent 评估集不存在: {path}（先运行 python scripts/build_data.py）"
        )
    rows = read_jsonl(path)
    details: list[dict[str, Any]] = []
    successes = 0
    total_steps = 0
    total_latency_ms = 0.0

    for row in rows:
        task = str(row.get("task", ""))
        keywords = [str(item) for item in row.get("expected_keywords", []) or []]
        expected_tools = [str(item) for item in row.get("expected_tools", []) or []]
        trace = agent.run(task)
        actual_tools = [
            step.action for step in trace.steps if step.action and step.action not in ("final", None)
        ]
        keyword_ok = any(keyword in trace.answer for keyword in keywords) if keywords else bool(trace.answer.strip())
        tools_ok = all(tool in actual_tools for tool in expected_tools)
        success = bool(trace.answer.strip()) and keyword_ok and tools_ok
        successes += int(success)
        total_steps += len(trace.steps)
        total_latency_ms += trace.elapsed_ms
        details.append(
            {
                "task": task,
                "success": success,
                "answer": trace.answer,
                "actions": actual_tools,
                "expected_tools": expected_tools,
                "num_steps": len(trace.steps),
                "elapsed_ms": trace.elapsed_ms,
            }
        )

    num = len(rows)
    metrics: dict[str, Any] = {
        "num_tasks": num,
        "success_rate": round(successes / num, 4) if num else 0.0,
        "avg_steps": round(total_steps / num, 3) if num else 0.0,
        "avg_latency_ms": round(total_latency_ms / num, 3) if num else 0.0,
        "details": details,
    }
    LOGGER.info("Agent 评估完成: tasks=%d success_rate=%.4f", num, metrics["success_rate"])
    return metrics


__all__ = ["evaluate_agent"]
