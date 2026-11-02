"""DQN 训练入口脚本。

用法::

    python scripts/train_dqn.py --config configs/default.yaml
    python scripts/train_dqn.py --config configs/quick.yaml
    python scripts/train_dqn.py --episodes 60        # 覆盖配置里的回合数

训练产物：``artifacts/rl/dqn.pt``；指标：``reports/rl_metrics.json``（dqn 节）与
``reports/rl_curve.json``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 允许直接以脚本方式运行（不依赖 pytest 的 pythonpath）
    sys.path.insert(0, str(ROOT))

from minillm.config import REPO_ROOT, ConfigError, load_config  # noqa: E402
from minillm.rl.train import run_dqn  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在 GridWorld 上训练 DQN 智能体")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径（相对仓库根目录），默认 configs/default.yaml",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="覆盖 cfg.dqn.episodes 的训练回合数（不传则使用配置文件里的值）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """加载配置、训练 DQN 并打印指标摘要。"""
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    try:
        cfg = load_config(config_path)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"[错误] 无法加载配置 {config_path}: {exc}", file=sys.stderr)
        return 2

    if args.episodes is not None:
        if args.episodes <= 0:
            print("[错误] --episodes 必须为正整数", file=sys.stderr)
            return 2
        cfg.dqn.episodes = args.episodes

    result = run_dqn(cfg)
    metrics = result["dqn"]
    print("=" * 66)
    print("DQN 训练完成（GridWorld 5x5）")
    print(f"  配置路径            : {config_path}")
    print(f"  训练回合数          : {metrics['episodes']}")
    print(f"  最终 greedy 成功率  : {metrics['final_success_rate']:.2f}"
          f"（{metrics['eval_episodes']} 个 greedy 回合）")
    print(f"  历史最优成功率      : {metrics['best_success_rate']:.2f}")
    print(f"  最近 20 回合平均回报: {metrics['avg_return_last20']:.4f}")
    print(f"  梯度更新次数        : {metrics['total_updates']}")
    print(f"  训练耗时            : {metrics['elapsed_sec']:.2f}s")
    print(f"  模型产物            : artifacts/rl/dqn.pt")
    print(f"  指标/曲线           : reports/rl_metrics.json, reports/rl_curve.json")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
