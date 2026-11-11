"""ReAct 执行循环：规划 → 调工具 → 记录 observation → 直到 final 或步数耗尽。

ReAct（Reason + Act）的思想是"边想边做"：每一步先产出 thought/action，
调用工具拿到 observation 再决定下一步。本模块只依赖 ``BasePlanner`` 与
``ToolRegistry`` 两个抽象，因此规则规划器和 LLM 规划器可以无缝替换（Strategy）。
"""

from __future__ import annotations

import time
from typing import Any

from minillm.agent.planner import BasePlanner
from minillm.agent.tools import ToolRegistry
from minillm.types import AgentStep, AgentTrace
from minillm.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _last_observation(steps: list[AgentStep]) -> str:
    """取最后一条非空 observation 作为兜底答案（压成单行）。"""
    for step in reversed(steps):
        text = (step.observation or "").strip()
        if text:
            return text.splitlines()[0][:300]
    return ""


class ReActAgent:
    """ReAct Agent：组合规划器与工具注册表，产出可解释的 ``AgentTrace``。

    Parameters
    ----------
    planner:
        规划器策略（``RuleBasedPlanner`` / ``LLMPlanner``）。
    tools:
        工具注册表；未知工具/工具异常都会被转成 observation 里的错误文本。
    max_steps:
        最大循环步数，防止规划器异常时死循环。
    """

    def __init__(self, planner: BasePlanner, tools: ToolRegistry | None = None, max_steps: int = 5) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须为正整数")
        self.planner = planner
        self.tools = tools if tools is not None else ToolRegistry()
        self.max_steps = int(max_steps)

    def run(self, task: str) -> AgentTrace:
        """执行任务，返回完整轨迹（含每一步 thought/action/observation）。"""
        started = time.perf_counter()
        task = (task or "").strip()
        trace = AgentTrace(task=task)
        if not task:
            trace.success = False
            trace.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return trace

        for step_no in range(1, self.max_steps + 1):
            try:
                decision = self.planner.next_action(task, trace.steps, self.tools)
            except Exception as exc:  # 规划器异常不能炸掉整条请求
                LOGGER.warning("规划器异常: %s", exc)
                decision = {
                    "thought": f"规划器异常: {exc}",
                    "action": "final",
                    "action_input": {"answer": _last_observation(trace.steps) or f"规划器异常: {exc}"},
                }

            if decision is None:
                # 规划器主动放弃（例如 LLM 输出无法解析）：用已有观察兜底结束
                answer = _last_observation(trace.steps) or "（规划器未能给出有效动作）"
                trace.steps.append(
                    AgentStep(step=step_no, thought="规划器未给出下一步动作，结束。", action=None,
                              action_input={}, observation=answer)
                )
                trace.answer = answer
                break

            thought = str(decision.get("thought") or "")
            action = decision.get("action")
            raw_input = decision.get("action_input")
            action_input: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}

            if action is None or action == "final":
                answer = str(action_input.get("answer") or "").strip()
                if not answer:
                    answer = _last_observation(trace.steps) or "（无可用答案）"
                trace.steps.append(
                    AgentStep(step=step_no, thought=thought or "结束。", action="final",
                              action_input={"answer": answer}, observation=answer)
                )
                trace.answer = answer
                break

            # 执行工具：失败也写进 observation，让规划器有机会换策略/继续
            result = self.tools.execute(str(action), action_input)
            observation = result.output if result.ok else f"工具执行失败[{action}]: {result.error}"
            trace.steps.append(
                AgentStep(
                    step=step_no,
                    thought=thought,
                    action=str(action),
                    action_input=action_input,
                    observation=observation,
                )
            )
        else:
            # for-else：步数耗尽仍未 final
            answer = _last_observation(trace.steps) or "任务未在限定步数内完成。"
            trace.answer = answer
            LOGGER.warning("Agent 达到最大步数 %d，任务: %s", self.max_steps, task)

        trace.success = bool(trace.answer.strip())
        trace.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return trace


__all__ = ["ReActAgent"]
