"""测试 ReAct Agent：算术/单位/知识检索/多步任务、错误注入、步数上限与评估指标。"""

from __future__ import annotations

import json
from pathlib import Path

from minillm.agent import (
    LLMPlanner,
    ReActAgent,
    RuleBasedPlanner,
    build_planner,
    build_tools,
)
from minillm.agent.tools import ToolRegistry
from minillm.config import default_config
from minillm.evaluation import evaluate_agent
from minillm.rag import build_rag_pipeline
from minillm.types import AgentStep

FAQ_ROWS = [
    {
        "id": "faq-001", "doc_id": "kb-01", "title": "第一食堂早餐时间",
        "question": "第一食堂早餐几点开始？", "answer": "第一食堂早餐开放时间为 7:00-9:00。",
        "tags": ["餐饮"], "split": "train",
    },
    {
        "id": "faq-002", "doc_id": "kb-02", "title": "图书馆开放时间",
        "question": "图书馆周五几点闭馆？", "answer": "图书馆周一至周五 8:00-22:00。",
        "tags": ["学习"], "split": "train",
    },
    {
        "id": "faq-003", "doc_id": "kb-03", "title": "宿舍报修电话",
        "question": "宿舍报修电话是多少？", "answer": "后勤报修电话 8260 1122。",
        "tags": ["后勤"], "split": "eval",
    },
]

KB_ROWS = [
    {
        "doc_id": "kb-01", "title": "校园餐饮服务", "summary": "第一食堂早餐开放时间 7:00-9:00。",
        "source": "后勤服务中心", "tags": ["校园", "餐饮"], "split": "train",
    },
    {
        "doc_id": "kb-02", "title": "图书馆服务", "summary": "图书馆周一至周五 8:00-22:00 开放。",
        "source": "图书馆", "tags": ["校园", "学习"], "split": "train",
    },
    {
        "doc_id": "kb-03", "title": "宿舍与后勤", "summary": "后勤报修电话 8260 1122。",
        "source": "后勤服务中心", "tags": ["校园", "后勤"], "split": "eval",
    },
]

AGENT_TASKS = [
    {"task": "计算 (125+75)*3/5 等于多少", "expected_keywords": ["120"], "expected_tools": ["calculator"]},
    {"task": "36 摄氏度等于多少华氏度？", "expected_keywords": ["96.8"], "expected_tools": ["unit_convert"]},
    {"task": "图书馆周五几点闭馆？", "expected_keywords": ["22:00"], "expected_tools": ["knowledge_search"]},
    {"task": "统计这段文字有多少个字：西安交通大学", "expected_keywords": ["6"], "expected_tools": ["text_stats"]},
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _demo_cfg(tmp_path: Path):
    """临时项目配置：有 raw/processed 数据，无训练产物（走 Extractive + 规则规划器）。"""
    cfg = default_config()
    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.reports_dir = str(tmp_path / "reports")
    cfg.serve.policy_ckpt = str(tmp_path / "artifacts" / "ppo" / "model.pt")
    cfg.serve.fallback_policy_ckpt = str(tmp_path / "artifacts" / "sft" / "model.pt")
    cfg.serve.tokenizer_path = str(tmp_path / "artifacts" / "tokenizer.json")
    cfg.serve.reward_ckpt = str(tmp_path / "artifacts" / "reward" / "model.pt")
    raw = tmp_path / "data" / "raw"
    _write_jsonl(raw / "faq.jsonl", FAQ_ROWS)
    _write_jsonl(raw / "kb_docs.jsonl", KB_ROWS)
    _write_jsonl(tmp_path / "data" / "processed" / "agent_tasks.jsonl", AGENT_TASKS)
    return cfg


def _build_agent(tmp_path: Path, max_steps: int | None = None) -> ReActAgent:
    cfg = _demo_cfg(tmp_path)
    pipeline = build_rag_pipeline(cfg)
    agent = ReActAgent(
        build_planner(cfg, pipeline),
        build_tools(cfg, pipeline),
        max_steps=max_steps or cfg.agent.max_steps,
    )
    return agent


# ---------------------------------------------------------------------- 规则规划器
def test_planner_prioritizes_calculator_then_answers(tmp_path: Path) -> None:
    cfg = _demo_cfg(tmp_path)
    pipeline = build_rag_pipeline(cfg)
    planner = RuleBasedPlanner()
    tools = build_tools(cfg, pipeline)

    decision = planner.next_action("计算 (125+75)*3/5 等于多少", [], tools)
    assert decision is not None and decision["action"] == "calculator"
    assert decision["action_input"]["expression"] == "(125+75)*3/5"

    steps = [AgentStep(step=1, thought="", action="calculator", action_input={}, observation="120")]
    final = planner.next_action("计算 (125+75)*3/5 等于多少", steps, tools)
    assert final is not None and final["action"] == "final"
    assert final["action_input"]["answer"] == "120"


def test_planner_does_not_misclassify_knowledge_questions(tmp_path: Path) -> None:
    """“统计学院”“GPA 怎么算”这类知识问题不能被误判为工具任务。"""
    cfg = _demo_cfg(tmp_path)
    pipeline = build_rag_pipeline(cfg)
    planner = RuleBasedPlanner()
    tools = build_tools(cfg, pipeline)
    for task in ("统计学院有多少学生？", "GPA 怎么算？"):
        decision = planner.next_action(task, [], tools)
        assert decision is not None
        assert decision["action"] == "knowledge_search", task


def test_build_planner_falls_back_to_rule_when_llm_unavailable(tmp_path: Path) -> None:
    cfg = _demo_cfg(tmp_path)
    cfg.agent.planner = "llm"
    planner = build_planner(cfg, None)
    assert isinstance(planner, RuleBasedPlanner)


def test_llm_planner_parse_json_actions() -> None:
    planner = LLMPlanner(generator=object())
    tool_action = planner._parse(
        '思考中 {"thought": "算一下", "action": "calculator", "action_input": {"expression": "1+1"}}'
    )
    assert tool_action is not None
    assert tool_action["action"] == "calculator"
    assert tool_action["action_input"] == {"expression": "1+1"}
    final_action = planner._parse('{"thought": "完成", "action": "final", "action_input": {"answer": "42"}}')
    assert final_action is not None and final_action["action_input"]["answer"] == "42"
    assert planner._parse("这不是 JSON") is None
    assert planner._parse('{"action": "final", "action_input": {}}') is None


# ---------------------------------------------------------------------- ReAct 任务
def test_calculator_task(tmp_path: Path) -> None:
    trace = _build_agent(tmp_path).run("计算 (125+75)*3/5 等于多少")
    assert trace.success is True
    assert trace.answer == "120"
    assert trace.steps[0].action == "calculator"
    assert trace.elapsed_ms >= 0


def test_unit_conversion_task(tmp_path: Path) -> None:
    trace = _build_agent(tmp_path).run("36 摄氏度等于多少华氏度？")
    assert trace.success is True
    assert "96.8" in trace.answer
    assert "unit_convert" in [step.action for step in trace.steps]


def test_knowledge_search_task_uses_answer_sentence(tmp_path: Path) -> None:
    trace = _build_agent(tmp_path).run("图书馆周五几点闭馆？")
    assert trace.success is True
    assert "22:00" in trace.answer
    assert trace.steps[0].action == "knowledge_search"


def test_multi_step_task_combines_tools(tmp_path: Path) -> None:
    trace = _build_agent(tmp_path).run("查询第一食堂早餐时间，并计算 7 + 2")
    actions = [step.action for step in trace.steps]
    assert actions[:2] == ["calculator", "knowledge_search"]
    assert "9" in trace.answer
    assert "7:00" in trace.answer
    assert trace.success is True


def test_tool_error_is_recorded_in_observation(tmp_path: Path) -> None:
    trace = _build_agent(tmp_path).run("计算 1/0 等于多少")
    calculator_step = trace.steps[0]
    assert calculator_step.action == "calculator"
    assert "工具执行失败" in calculator_step.observation
    assert "除零" in calculator_step.observation


def test_unknown_tool_registry_ends_with_final_answer(tmp_path: Path) -> None:
    cfg = _demo_cfg(tmp_path)
    planner = RuleBasedPlanner()
    empty_tools = ToolRegistry()
    agent = ReActAgent(planner, empty_tools, max_steps=3)
    trace = agent.run("计算 1+1 等于多少")
    assert trace.steps[-1].action == "final"
    assert trace.answer.strip()
    assert trace.success is True


class _AlwaysCalculatorPlanner:
    """始终要求调用计算器的规划器，用于验证 max_steps 兜底。"""

    def next_action(self, task, steps, tools):  # noqa: ANN001
        return {"thought": "继续算", "action": "calculator", "action_input": {"expression": "1+1"}}


class _NonePlanner:
    """模拟 LLM 输出无法解析的规划器。"""

    def next_action(self, task, steps, tools):  # noqa: ANN001
        return None


class _RaisingPlanner:
    """模拟规划器内部异常。"""

    def next_action(self, task, steps, tools):  # noqa: ANN001
        raise RuntimeError("simulated planner failure")


def test_max_steps_limits_loop(tmp_path: Path) -> None:
    cfg = _demo_cfg(tmp_path)
    tools = build_tools(cfg, build_rag_pipeline(cfg))
    trace = ReActAgent(_AlwaysCalculatorPlanner(), tools, max_steps=2).run("1+1")
    assert [step.action for step in trace.steps] == ["calculator", "calculator"]
    assert trace.answer == "2"
    assert trace.success is True


def test_planner_returning_none_ends_gracefully(tmp_path: Path) -> None:
    cfg = _demo_cfg(tmp_path)
    tools = build_tools(cfg, build_rag_pipeline(cfg))
    trace = ReActAgent(_NonePlanner(), tools, max_steps=3).run("随便问点什么")
    assert len(trace.steps) == 1
    assert trace.answer.strip()
    assert trace.success is True


def test_planner_exception_does_not_crash(tmp_path: Path) -> None:
    cfg = _demo_cfg(tmp_path)
    tools = build_tools(cfg, build_rag_pipeline(cfg))
    trace = ReActAgent(_RaisingPlanner(), tools, max_steps=3).run("随便问点什么")
    assert trace.steps[-1].action == "final"
    assert "规划器异常" in trace.answer


def test_empty_task_returns_unsuccessful_trace(tmp_path: Path) -> None:
    trace = _build_agent(tmp_path).run("   ")
    assert trace.success is False
    assert trace.steps == []


# ---------------------------------------------------------------------- Agent 评估
def test_evaluate_agent_metrics(tmp_path: Path) -> None:
    cfg = _demo_cfg(tmp_path)
    agent = _build_agent(tmp_path)
    metrics = evaluate_agent(agent, cfg.processed_dir() / "agent_tasks.jsonl")

    assert set(metrics) == {"num_tasks", "success_rate", "avg_steps", "avg_latency_ms", "details"}
    assert metrics["num_tasks"] == len(AGENT_TASKS)
    assert metrics["success_rate"] == 1.0
    assert metrics["avg_steps"] > 0
    assert len(metrics["details"]) == len(AGENT_TASKS)
    assert all("actions" in item for item in metrics["details"])
