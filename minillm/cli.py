"""MiniLLM Studio 命令行入口（``python -m minillm <command>``）。

子命令
------
- ``data``                   调用 ``scripts/build_data.py`` 生成训练/评估数据
- ``train {pretrain,sft,reward,ppo,dqn}``  延迟 import 对应训练器并运行
- ``eval {rag,agent,all}``   运行离线评估，指标写入 ``reports/app_metrics.json``
- ``rag ask "问题"``          单问 RAG 检索增强问答
- ``agent run "任务"``       单次 ReAct Agent 任务
- ``serve``                  启动 FastAPI 服务
- ``smoke``                  不依赖训练产物的端到端冒烟（RAG 回答 + 计算器 Agent）

训练器采用**延迟 import**，因此没装 torch / 训练模块未生成时，其余命令仍可用。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from minillm import __version__
from minillm.config import REPO_ROOT, ConfigError, ProjectConfig, default_config, load_config
from minillm.utils.logging import get_logger

LOGGER = get_logger(__name__)
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "default.yaml"

# 训练阶段 -> 候选 (模块, 函数) 列表；兼容训练模块不同的导出方式
_TRAIN_TARGETS: dict[str, list[tuple[str, str]]] = {
    "pretrain": [("minillm.train", "run_pretrain"), ("minillm.train.pretrain", "run_pretrain")],
    "sft": [("minillm.train", "run_sft"), ("minillm.train.sft", "run_sft")],
    "reward": [("minillm.train", "run_reward"), ("minillm.train.reward", "run_reward")],
    "ppo": [("minillm.train", "run_ppo"), ("minillm.train.ppo", "run_ppo")],
    "dqn": [("minillm.rl.train", "run_dqn"), ("minillm.rl", "run_dqn")],
}


# ---------------------------------------------------------------------- 公共
def resolve_config(path: str | Path | None = None) -> ProjectConfig:
    """加载配置：显式路径 > 环境变量 ``MINILLM_CONFIG`` > 默认配置文件 > 内置默认值。"""
    if path:
        return load_config(path)
    env_path = os.environ.get("MINILLM_CONFIG")
    if env_path:
        return load_config(env_path)
    if DEFAULT_CONFIG_PATH.exists():
        return load_config(DEFAULT_CONFIG_PATH)
    return default_config()


def _print_json(data: Any) -> None:
    """统一 UTF-8 友好的 JSON 打印。"""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _resolve_callable(candidates: list[tuple[str, str]]) -> Callable[..., Any] | None:
    """按候选列表寻找可调用的训练函数（延迟 import，失败返回 None）。"""
    for module_name, func_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        func = getattr(module, func_name, None)
        if callable(func):
            return func
    return None


# ---------------------------------------------------------------------- 子命令
def cmd_data(cfg: ProjectConfig) -> int:
    """调用数据构建脚本（脚本会覆盖 ``<processed_dir>`` 下的产物）。"""
    script = REPO_ROOT / "scripts" / "build_data.py"
    if not script.exists():
        print(f"[minillm] 找不到数据构建脚本: {script}", file=sys.stderr)
        return 1
    command = [sys.executable, str(script), "--out", str(cfg.processed_dir())]
    LOGGER.info("运行数据构建: %s", " ".join(command))
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


def cmd_train(cfg: ProjectConfig, stage: str) -> int:
    """运行指定训练阶段（延迟 import 对应 ``run_*``）。"""
    func = _resolve_callable(_TRAIN_TARGETS[stage])
    if func is None:
        print(
            f"[minillm] 训练阶段 {stage!r} 暂不可用：训练模块未实现或依赖缺失（如 torch 未安装）",
            file=sys.stderr,
        )
        return 1
    LOGGER.info("开始训练阶段: %s", stage)
    metrics = func(cfg)
    _print_json(metrics)
    return 0


def cmd_eval(cfg: ProjectConfig, target: str) -> int:
    """运行离线评估并合并写入 ``reports/app_metrics.json``。

    RAG 同时评估两组：
    - ``rag``：``data/processed/eval_qa.jsonl``（FAQ 问法改写，指标偏乐观）；
    - ``rag_hard``：内置的语义改写难例（人工构造，暴露纯词法检索的不足）。
    """
    from minillm.agent import ReActAgent, build_planner, build_tools
    from minillm.evaluation import HARD_EVAL_QUERIES, evaluate_agent, evaluate_rag
    from minillm.rag import build_rag_pipeline
    from minillm.utils.metrics import save_metrics

    reports_path = cfg.path("reports_dir") / "app_metrics.json"
    results: dict[str, Any] = {}
    pipeline = None
    if target in ("rag", "all"):
        pipeline = build_rag_pipeline(cfg)
        rag_metrics = evaluate_rag(pipeline, cfg.processed_dir() / "eval_qa.jsonl", top_k=cfg.rag.top_k)
        save_metrics(reports_path, "rag", rag_metrics)
        results["rag"] = rag_metrics

        hard_metrics = evaluate_rag(pipeline, HARD_EVAL_QUERIES, top_k=cfg.rag.top_k)
        hard_payload = {
            **hard_metrics,
            "note": "人工构造的语义改写难例（非训练数据；gold 来自 eval 划分文档 kb-campus-07/10）",
        }
        save_metrics(reports_path, "rag_hard", hard_payload)
        results["rag_hard"] = hard_payload
    if target in ("agent", "all"):
        if pipeline is None:
            pipeline = build_rag_pipeline(cfg)
        tools = build_tools(cfg, pipeline)
        planner = build_planner(cfg, pipeline)
        agent = ReActAgent(planner, tools, max_steps=cfg.agent.max_steps)
        agent_metrics = evaluate_agent(agent, cfg.processed_dir() / "agent_tasks.jsonl")
        save_metrics(reports_path, "agent", agent_metrics)
        results["agent"] = agent_metrics
    _print_json(results)
    return 0


def cmd_rag_ask(cfg: ProjectConfig, question: str, top_k: int | None = None) -> int:
    """RAG 单问：检索 + 生成 + 引用。"""
    from minillm.rag import build_rag_pipeline

    if top_k is not None:
        if top_k <= 0:
            print("[minillm] --top-k 必须为正整数", file=sys.stderr)
            return 2
        cfg.rag.top_k = int(top_k)
    pipeline = build_rag_pipeline(cfg)
    answer = pipeline.answer(question)
    _print_json(answer.to_dict())
    return 0


def cmd_agent_run(cfg: ProjectConfig, task: str, max_steps: int | None = None) -> int:
    """执行单次 ReAct Agent 任务并打印完整轨迹。"""
    from minillm.agent import ReActAgent, build_planner, build_tools
    from minillm.rag import build_rag_pipeline

    pipeline = build_rag_pipeline(cfg)
    steps = int(max_steps or cfg.agent.max_steps)
    if steps <= 0:
        print("[minillm] --max-steps 必须为正整数", file=sys.stderr)
        return 2
    agent = ReActAgent(build_planner(cfg, pipeline), build_tools(cfg, pipeline), max_steps=steps)
    _print_json(agent.run(task).to_dict())
    return 0


def cmd_serve(cfg: ProjectConfig, host: str | None = None, port: int | None = None) -> int:
    """启动 FastAPI 服务（阻塞，直到 Ctrl+C）。"""
    import uvicorn

    from minillm.serving.app import create_app

    app = create_app(cfg)
    use_host = host or cfg.serve.host
    use_port = int(port or cfg.serve.port)
    LOGGER.info("启动服务: http://%s:%s", use_host, use_port)
    uvicorn.run(app, host=use_host, port=use_port, log_level=cfg.serve.log_level)
    return 0


def run_smoke(cfg: ProjectConfig) -> dict[str, Any]:
    """端到端冒烟：RAG 回答一个 FAQ 问题 + 跑一个计算器 Agent 任务。

    不依赖任何训练产物（无 checkpoint 时走 Extractive 回退），目标 < 20 秒。
    """
    from minillm.agent import ReActAgent, build_planner, build_tools
    from minillm.rag import build_rag_pipeline

    started = time.perf_counter()
    pipeline = build_rag_pipeline(cfg)
    # 从实际加载的 FAQ 里取一个问题，保证 smoke 在任意数据/临时 fixture 下都能跑
    question = next(
        (doc.content.splitlines()[0] for doc in pipeline.docs if doc.source == "faq" and doc.content),
        "第一食堂早餐几点开始？",
    )
    rag_answer = pipeline.answer(question)
    agent_task = "计算 12 * (3 + 4) 等于多少"
    agent = ReActAgent(
        build_planner(cfg, pipeline),
        build_tools(cfg, pipeline),
        max_steps=cfg.agent.max_steps,
    )
    trace = agent.run(agent_task)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return {
        "version": __version__,
        "ok": bool(rag_answer.answer.strip()) and trace.success,
        "generator": pipeline.generator_name,
        "rag": {
            "question": question,
            "answer": rag_answer.answer,
            "source": rag_answer.source,
            "num_citations": len(rag_answer.citations),
        },
        "agent": {
            "task": agent_task,
            "answer": trace.answer,
            "success": trace.success,
            "actions": [step.action for step in trace.steps],
        },
        "elapsed_ms": elapsed_ms,
    }


def cmd_smoke(cfg: ProjectConfig) -> int:
    """运行冒烟并打印 JSON；成功返回 0。"""
    result = run_smoke(cfg)
    _print_json(result)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------- 解析器
def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器（独立函数便于测试命令行结构）。"""
    parser = argparse.ArgumentParser(
        prog="minillm",
        description="MiniLLM Studio：轻量级 RLHF + RAG + Agent 全链路实践平台",
        epilog="示例：python -m minillm smoke | python -m minillm rag ask \"食堂几点开门？\"",
    )
    sub = parser.add_subparsers(dest="command", metavar="{data,train,eval,rag,agent,serve,smoke}")

    data_parser = sub.add_parser("data", help="生成训练/评估数据")
    data_parser.add_argument("--config", default=None, help="配置文件路径")

    train_parser = sub.add_parser("train", help="运行训练阶段")
    train_parser.add_argument("stage", choices=["pretrain", "sft", "reward", "ppo", "dqn"])
    train_parser.add_argument("--config", default=None, help="配置文件路径")

    eval_parser = sub.add_parser("eval", help="运行离线评估")
    eval_parser.add_argument("target", choices=["rag", "agent", "all"])
    eval_parser.add_argument("--config", default=None, help="配置文件路径")

    rag_parser = sub.add_parser("rag", help="RAG 检索/问答")
    rag_sub = rag_parser.add_subparsers(dest="rag_command", metavar="{ask}")
    rag_ask = rag_sub.add_parser("ask", help="提问并返回带引用的答案")
    rag_ask.add_argument("question", help="自然语言问题")
    rag_ask.add_argument("--top-k", type=int, default=None, help="检索条数")
    rag_ask.add_argument("--config", default=None, help="配置文件路径")

    agent_parser = sub.add_parser("agent", help="ReAct Agent")
    agent_sub = agent_parser.add_subparsers(dest="agent_command", metavar="{run}")
    agent_run = agent_sub.add_parser("run", help="执行一个任务并打印轨迹")
    agent_run.add_argument("task", help="自然语言任务")
    agent_run.add_argument("--max-steps", type=int, default=None, help="最大步数")
    agent_run.add_argument("--config", default=None, help="配置文件路径")

    serve_parser = sub.add_parser("serve", help="启动 FastAPI 服务")
    serve_parser.add_argument("--config", default=None, help="配置文件路径")
    serve_parser.add_argument("--host", default=None, help="监听地址")
    serve_parser.add_argument("--port", type=int, default=None, help="监听端口")

    smoke_parser = sub.add_parser("smoke", help="不依赖 checkpoint 的端到端冒烟")
    smoke_parser.add_argument("--config", default=None, help="配置文件路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主函数：解析参数并分发；返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "rag" and getattr(args, "rag_command", None) is None:
        print('用法: python -m minillm rag ask "问题" [--top-k N] [--config PATH]', file=sys.stderr)
        return 2
    if args.command == "agent" and getattr(args, "agent_command", None) is None:
        print('用法: python -m minillm agent run "任务" [--max-steps N] [--config PATH]', file=sys.stderr)
        return 2
    try:
        cfg = resolve_config(getattr(args, "config", None))
        if args.command == "data":
            return cmd_data(cfg)
        if args.command == "train":
            return cmd_train(cfg, args.stage)
        if args.command == "eval":
            return cmd_eval(cfg, args.target)
        if args.command == "rag":
            return cmd_rag_ask(cfg, args.question, top_k=args.top_k)
        if args.command == "agent":
            return cmd_agent_run(cfg, args.task, max_steps=args.max_steps)
        if args.command == "serve":
            return cmd_serve(cfg, host=args.host, port=args.port)
        if args.command == "smoke":
            return cmd_smoke(cfg)
        print(f"[minillm] 未知命令: {args.command}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[minillm] 缺少数据/文件: {exc}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"[minillm] 配置错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - 交互式中断
        print("\n[minillm] 已中断", file=sys.stderr)
        return 130
    except Exception as exc:  # 兜底：CLI 不打印 traceback 污染输出，但保留错误摘要
        LOGGER.error("命令执行失败: %s", exc)
        return 1


__all__ = [
    "build_parser",
    "cmd_agent_run",
    "cmd_data",
    "cmd_eval",
    "cmd_rag_ask",
    "cmd_serve",
    "cmd_smoke",
    "cmd_train",
    "main",
    "resolve_config",
    "run_smoke",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
