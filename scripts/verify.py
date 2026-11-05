"""一键验收脚本：数据构建 → 训练链路 → 评估 → 服务冒烟 → 单元测试。

用法
----
    python scripts/verify.py                # 全量验收（默认配置，CPU 约 5~8 分钟）
    python scripts/verify.py --quick        # 冒烟验收（configs/quick.yaml，约 1~2 分钟）
    python scripts/verify.py --skip-train   # 复用已有 artifacts，只跑评估/测试

每个步骤独立执行，失败不会中断后续步骤；最终输出 PASS/FAIL 汇总并写入
``reports/verify_report.json``，退出码 0 表示全部通过。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "verify_report.json"


def run_step(name: str, cmd: list[str], timeout: int = 900) -> dict:
    """执行一个子步骤，返回 {name, cmd, ok, elapsed_sec, tail}。"""
    print(f"\n===== [{name}] {' '.join(cmd)} =====", flush=True)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        tail = "\n".join(output.strip().splitlines()[-25:])
    except subprocess.TimeoutExpired as exc:
        ok = False
        tail = f"超时（>{timeout}s）: {exc}"
    elapsed = round(time.perf_counter() - started, 2)
    print(tail, flush=True)
    print(f"----- [{name}] {'PASS' if ok else 'FAIL'} ({elapsed}s) -----", flush=True)
    return {"name": name, "cmd": cmd, "ok": ok, "elapsed_sec": elapsed, "tail": tail}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MiniLLM Studio 一键验收")
    parser.add_argument("--config", default="configs/default.yaml", help="训练配置")
    parser.add_argument("--quick", action="store_true", help="使用 configs/quick.yaml 冒烟")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，复用已有 artifacts")
    parser.add_argument("--skip-tests", action="store_true", help="跳过 pytest")
    args = parser.parse_args(argv)
    config = "configs/quick.yaml" if args.quick else args.config

    py = sys.executable
    steps: list[tuple[str, list[str]]] = [
        ("构建数据", [py, "scripts/build_data.py"]),
    ]
    if not args.skip_train:
        steps += [
            ("预训练", [py, "scripts/train_pretrain.py", "--config", config]),
            ("SFT 微调", [py, "scripts/train_sft.py", "--config", config]),
            ("奖励模型", [py, "scripts/train_reward.py", "--config", config]),
            ("PPO/RLHF", [py, "scripts/train_ppo.py", "--config", config]),
            ("DQN 强化学习", [py, "scripts/train_dqn.py", "--config", config]),
        ]
    steps += [
        ("RAG 评估", [py, "scripts/eval_rag.py", "--config", config]),
        ("Agent 评估", [py, "scripts/eval_agent.py", "--config", config]),
        ("CLI/服务冒烟", [py, "-m", "minillm", "smoke"]),
    ]
    if not args.skip_tests:
        steps.append(("单元测试", [py, "-m", "pytest", "tests", "-q"]))

    results = [run_step(name, cmd) for name, cmd in steps]

    # 收集关键指标汇总
    metrics: dict = {}
    for path in (ROOT / "reports").glob("*_metrics.json"):
        try:
            metrics[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

    summary = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "steps": results,
        "metrics_files": sorted(metrics),
        "passed": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "all_passed": all(r["ok"] for r in results),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ 验收汇总 ================")
    for result in results:
        mark = "PASS" if result["ok"] else "FAIL"
        print(f"[{mark}] {result['name']:<14} {result['elapsed_sec']:>7.2f}s")
    print(f"通过 {summary['passed']} / 失败 {summary['failed']}；报告: {REPORT}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
