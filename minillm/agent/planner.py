"""规划器：决定 Agent 下一步调用哪个工具（Strategy 模式）。

- ``RuleBasedPlanner``：**必做**的确定性规划器。用规则识别任务意图（算式 / 单位换算 /
  文本统计 / 知识检索），按优先级逐步执行，最后基于 observation 汇总答案；通过
  "已执行过的工具不重复调用" 保证不会陷入死循环。
- ``LLMPlanner``：可选实现，让 TinyGPT 输出 JSON 动作；解析失败返回 ``None``，
  由 ``ReActAgent`` 结束循环（上层兜底），从而不会把模型幻觉变成崩溃。
- ``build_planner``：按 ``cfg.agent.planner`` 构造；``llm`` 不可用时告警并回退 ``rule``。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from minillm.agent.tools import (
    ToolRegistry,
    extract_math_expression,
    find_supported_units,
)
from minillm.rag.pipeline import TinyGPTGenerator
from minillm.types import AgentStep
from minillm.utils.logging import get_logger

LOGGER = get_logger(__name__)

# 意图识别关键词
_MATH_HINT_RE = re.compile(r"\d\s*[+\-*/%]\s*[\d(]")
_STATS_PRIMARY = ("字数", "字符数", "词数", "行数", "多少字")
_STATS_CONTEXT = ("这段", "这句", "文本", "：", ":")
_UNIT_HINT_KEYWORDS = ("换算", "转换", "换成", "等于多少", "是多少")
_KNOWLEDGE_HINTS = (
    "查询", "查一下", "查找", "搜索", "检索", "请问", "告诉我", "介绍",
    "是什么", "什么是", "有哪些", "怎么", "如何", "哪里", "几点", "什么时候",
)
_QUERY_PREFIXES = ("请", "帮我", "帮忙", "查询", "查一下", "查找", "搜索", "检索", "告诉我", "请问")
_CALC_INSTRUCTION_RE = re.compile(r"(?:并|然后|接着|再)?\s*(?:请)?\s*(?:计算|算一下|算出)[^。？?]*")
_MEANINGFUL_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def _concat_unique(parts: list[str]) -> str:
    """保序去重后拼接。"""
    seen: list[str] = []
    for part in parts:
        part = part.strip()
        if part and part not in seen:
            seen.append(part)
    return "；".join(seen)


def _observation_to_sentence(observation: str) -> str:
    """把工具 observation 压成一句可读答案。

    知识检索输出形如 ``[1] doc / 标题（score=1.2）：正文…``，取"）："后的句子；
    FAQ 片段常以"问题\\n答案"结构出现，因此跳过疑问句、取真正的答案句；
    计算器/单位换算等单行输出原样保留。
    """
    lines = [ln.strip() for ln in (observation or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    if "）：" in first:
        first = first.split("）：", 1)[1].strip()
    # 首行是问句时，答案通常在下一行
    if first.endswith(("？", "?")) and len(lines) > 1:
        first = " ".join([first, *lines[1:]])
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?])", first) if s.strip()]
    while len(sentences) > 1 and sentences[0].endswith(("？", "?")):
        sentences.pop(0)  # 跳过"问题句"，取后面的答案句
    text = sentences[0] if sentences else first
    return text[:200]


class BasePlanner(ABC):
    """规划器策略接口（Template：ReAct 循环只依赖这个接口）。"""

    @abstractmethod
    def next_action(self, task: str, steps: list[AgentStep], tools: ToolRegistry) -> dict[str, Any] | None:
        """返回下一步动作 ``{"thought", "action", "action_input"}``。

        - ``action`` 为工具名：``action_input`` 是工具参数 dict；
        - ``action`` 为 ``None`` 或 ``"final"`` 表示结束，最终答案在 ``action_input["answer"]``；
        - 返回 ``None`` 表示规划器无法给出动作（上层结束循环并兜底）。
        """


class RuleBasedPlanner(BasePlanner):
    """规则规划器：意图优先级 calculator → unit_convert → text_stats → knowledge_search。"""

    # ------------------------------------------------------------------ 意图
    @staticmethod
    def _looks_like_math(task: str) -> bool:
        if _MATH_HINT_RE.search(task):
            return True
        return ("计算" in task or "算一下" in task) and any(ch.isdigit() for ch in task)

    @staticmethod
    def _looks_like_unit(task: str) -> bool:
        has_unit = bool(find_supported_units(task))
        has_hint = any(keyword in task for keyword in _UNIT_HINT_KEYWORDS)
        return has_unit and has_hint

    @staticmethod
    def _looks_like_stats(task: str) -> bool:
        """统计意图：明确的字数/词数关键词，或"统计 + 具体文本"（避免"统计学院"误判）。"""
        if any(keyword in task for keyword in _STATS_PRIMARY):
            return True
        return "统计" in task and any(context in task for context in _STATS_CONTEXT)

    @staticmethod
    def _wants_knowledge(task: str) -> bool:
        """混合任务里是否还包含"查资料"诉求（纯知识任务由主流程直接判定）。"""
        return any(hint in task for hint in _KNOWLEDGE_HINTS)

    def _intents(self, task: str, tools: ToolRegistry) -> list[str]:
        """按优先级给出本任务需要执行的工具动作序列。"""
        intents: list[str] = []
        math_expr = extract_math_expression(task)
        if math_expr and "calculator" in tools and self._looks_like_math(task):
            intents.append("calculator")
        if self._looks_like_unit(task) and "unit_convert" in tools:
            intents.append("unit_convert")
        if self._looks_like_stats(task) and "text_stats" in tools:
            intents.append("text_stats")
        # 纯知识任务（没有其它工具意图）默认检索；混合任务仅在出现检索/提问关键词时追加
        if "knowledge_search" in tools and (not intents or self._wants_knowledge(task)):
            intents.append("knowledge_search")
        return intents

    # ------------------------------------------------------------------ 参数
    @staticmethod
    def _unit_payload(task: str) -> dict[str, Any]:
        """单位换算工具直接吃自然语言 ``text``，解析逻辑收敛在工具内部。"""
        return {"text": task}

    @staticmethod
    def _stats_payload(task: str) -> dict[str, Any]:
        """抽取待统计文本：优先引号内容，其次冒号后的内容，最后去掉指令词。"""
        quoted = re.search(r"[\"“「『']([^\"”」』']+)[\"”」』']", task)
        if quoted:
            return {"text": quoted.group(1).strip()}
        if "：" in task or ":" in task:
            tail = re.split(r"[:：]", task, maxsplit=1)[1].strip()
            if tail:
                return {"text": tail}
        cleaned = task
        for keyword in (*_STATS_PRIMARY, "统计"):
            cleaned = cleaned.replace(keyword, " ")
        cleaned = re.sub(r"(请|帮我|这段|这句话|文本|一下|的|是多少)", " ", cleaned)
        cleaned = cleaned.strip(" ：:，,。？?")
        return {"text": cleaned or task}

    @staticmethod
    def _knowledge_query(task: str) -> str:
        """把任务改写成适合 BM25 的检索 query。"""
        query = task
        expr = extract_math_expression(task)
        if expr:
            query = query.replace(expr, " ")
        query = _CALC_INSTRUCTION_RE.sub(" ", query)
        for prefix in _QUERY_PREFIXES:  # 去掉口语化指令前缀
            while query.startswith(prefix):
                query = query[len(prefix):].lstrip(" ：:，,")
        query = re.sub(r"[？?。！!，,；;：:\s]+$", "", query).strip()
        return query or task.strip()

    # ------------------------------------------------------------------ 主流程
    def next_action(self, task: str, steps: list[AgentStep], tools: ToolRegistry) -> dict[str, Any] | None:
        task = (task or "").strip()
        used = {step.action for step in steps if step.action and step.action not in ("final", None)}
        intents = self._intents(task, tools)
        pending = [intent for intent in intents if intent not in used]

        if pending:
            action = pending[0]
            if action == "calculator":
                expression = extract_math_expression(task) or ""
                return {
                    "thought": f"任务包含算式 {expression!r}，先用计算器求解。",
                    "action": "calculator",
                    "action_input": {"expression": expression},
                }
            if action == "unit_convert":
                return {
                    "thought": "任务包含单位换算，调用单位换算工具。",
                    "action": "unit_convert",
                    "action_input": self._unit_payload(task),
                }
            if action == "text_stats":
                payload = self._stats_payload(task)
                return {
                    "thought": "任务要求统计文本信息，调用文本统计工具。",
                    "action": "text_stats",
                    "action_input": payload,
                }
            return {
                "thought": "需要事实性知识，先检索本地知识库。",
                "action": "knowledge_search",
                "action_input": {"query": self._knowledge_query(task)},
            }

        # 没有待执行动作：汇总 observation 作为最终答案
        observations = [step.observation for step in steps if step.observation]
        answer = _concat_unique([_observation_to_sentence(obs) for obs in observations])
        if not answer:
            answer = "抱歉，我无法从现有工具完成该任务。"
        return {
            "thought": "已获得所需信息，整理最终答案。",
            "action": "final",
            "action_input": {"answer": answer},
        }


class LLMPlanner(BasePlanner):
    """可选的 LLM 规划器：用 TinyGPT 生成 JSON 动作，解析失败返回 ``None``。

    说明：TinyGPT 只有约百万级参数，规划能力有限；这里不追求效果，而是演示
    "模型规划 + 规则兜底" 的工程闭环。构造前需确保检查点可用。
    """

    _ACTION_RE = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, generator: Any, max_new_tokens: int = 96) -> None:
        self.generator = generator
        self.max_new_tokens = int(max_new_tokens)

    def _build_prompt(self, task: str, steps: list[AgentStep], tools: ToolRegistry) -> str:
        tool_desc = "\n".join(f"- {item['name']}: {item['description']}" for item in tools.describe())
        history = "\n".join(
            f"第{item.step}步: 动作={item.action}, 输入={json.dumps(item.action_input, ensure_ascii=False)}, 观察={item.observation[:120]}"
            for item in steps
        ) or "（无）"
        return (
            "你是一个 ReAct 规划器，请只输出一行 JSON，不要输出其它内容。\n"
            f"可用工具：\n{tool_desc}\n"
            "输出格式：{\"thought\": \"思考\", \"action\": \"工具名或final\", \"action_input\": {...}}\n"
            f"任务：{task}\n历史：\n{history}\n"
            "如果信息已足够，用 action=final 且 action_input={\"answer\": \"最终答案\"}。\nJSON："
        )

    @classmethod
    def _parse(cls, text: str) -> dict[str, Any] | None:
        """从模型输出里提取第一个合法 JSON 动作。"""
        match = cls._ACTION_RE.search(text or "")
        if not match:
            return None
        try:
            data, _ = json.JSONDecoder().raw_decode(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        thought = str(data.get("thought") or "").strip()
        action = data.get("action")
        action_input = data.get("action_input")
        if action is not None:
            action = str(action)
        if not isinstance(action_input, dict):
            action_input = {}
        if action is None or action == "final":
            answer = str(action_input.get("answer") or "").strip()
            if not answer:
                return None
            return {"thought": thought or "模型判断可以结束。", "action": "final", "action_input": {"answer": answer}}
        return {"thought": thought or f"调用工具 {action}。", "action": action, "action_input": action_input}

    def next_action(self, task: str, steps: list[AgentStep], tools: ToolRegistry) -> dict[str, Any] | None:
        prompt = self._build_prompt(task, steps, tools)
        try:
            result = self.generator.generate(prompt, max_new_tokens=self.max_new_tokens, context="")
        except Exception as exc:  # 模型不可用时由上层兜底
            LOGGER.warning("LLMPlanner 生成失败: %s", exc)
            return None
        return self._parse(getattr(result, "text", ""))


def build_planner(cfg: Any, rag_pipeline: Any = None) -> BasePlanner:
    """按配置构造规划器；``llm`` 不可用时告警并回退 ``rule``。

    ``rag_pipeline`` 参数预留给未来 hybrid planner，当前规则规划器通过工具注册表
    间接访问 RAGPipeline，无需直接持有。
    """
    name = getattr(cfg.agent, "planner", "rule")
    if name == "llm":
        try:
            if not TinyGPTGenerator.is_ready(cfg):
                raise RuntimeError("未发现可用检查点（artifacts/ppo|sft/model.pt 与 tokenizer.json）")
            generator = TinyGPTGenerator(
                cfg,
                max_new_tokens=max(96, cfg.serve.max_new_tokens),
                temperature=cfg.serve.temperature,
                top_k=cfg.serve.top_k,
            )
            generator.load()
            LOGGER.info("Agent 规划器: LLMPlanner（TinyGPT 适配）")
            return LLMPlanner(generator)
        except Exception as exc:
            LOGGER.warning("LLMPlanner 不可用（%s），回退 RuleBasedPlanner", exc)
    return RuleBasedPlanner()


__all__ = ["BasePlanner", "LLMPlanner", "RuleBasedPlanner", "build_planner"]
