"""数据构建脚本：把 ``data/raw`` 的种子数据加工成训练/评估所需的 ``data/processed``。

数据来源
--------
- ``data/raw/faq.jsonl``      结构化 FAQ（问题-答案对 + 归属文档 + train/eval 划分）
- ``data/raw/kb_docs.jsonl``  知识库文档摘要（检索语料的"骨架"）
- ``data/raw/agent_tasks.jsonl`` Agent 评估任务

产物
----
- ``corpus.txt``        预训练语料（仅 train 划分，避免评估泄漏）
- ``sft.jsonl``         指令微调样本（每个 FAQ 生成 3 种问法）
- ``pref.jsonl``        偏好数据（chosen=正确答案，rejected=其他文档的答案）
- ``eval_qa.jsonl``     RAG 评估集（仅 eval 划分，含 gold 文档与关键词）
- ``agent_tasks.jsonl`` Agent 评估集
- ``stats.json``        数据统计

设计说明：用模板对问法做轻度改写（paraphrase），让模型/检索器见过多种表达，
同时保证 eval 文档整体不进入训练集，指标才有参考意义。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少原始数据文件: {path}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {lineno} 行不是合法 JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_question(question: str) -> str:
    return question.strip().rstrip("？?").strip()


def paraphrase(question: str) -> list[str]:
    """对问题做 3 种轻量改写，模拟用户的不同问法。"""
    base = _strip_question(question)
    return [f"{base}？", f"请问{base}？", f"我想知道{base}。"]


def extract_keywords(answer: str) -> list[str]:
    """抽取答案中的数字/英文关键词，用于 RAG 答案覆盖率等辅助评估。"""
    tokens = re.findall(r"\d+(?:[:：]\d+)?(?:\.\d+)?|[A-Za-z][A-Za-z0-9./_-]*", answer)
    return list(dict.fromkeys(tokens))


def _validate(faq: list[dict[str, Any]], docs: list[dict[str, Any]]) -> None:
    doc_ids = {d["doc_id"] for d in docs}
    seen: set[str] = set()
    for entry in faq:
        for key in ("id", "doc_id", "question", "answer", "split"):
            if key not in entry:
                raise ValueError(f"FAQ 缺少字段 {key}: {entry}")
        if entry["id"] in seen:
            raise ValueError(f"FAQ id 重复: {entry['id']}")
        seen.add(entry["id"])
        if entry["doc_id"] not in doc_ids:
            raise ValueError(f"FAQ {entry['id']} 引用了不存在的 doc_id: {entry['doc_id']}")
        if entry["split"] not in ("train", "eval"):
            raise ValueError(f"FAQ {entry['id']} 的 split 非法: {entry['split']}")
    for doc in docs:
        if doc.get("split") not in ("train", "eval"):
            raise ValueError(f"文档 {doc.get('doc_id')} 的 split 非法")


def build(faq_path: Path | None = None, docs_path: Path | None = None,
          tasks_path: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    """执行数据构建，返回统计信息。"""
    faq_path = faq_path or RAW_DIR / "faq.jsonl"
    docs_path = docs_path or RAW_DIR / "kb_docs.jsonl"
    tasks_path = tasks_path or RAW_DIR / "agent_tasks.jsonl"
    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    faq = read_jsonl(faq_path)
    docs = read_jsonl(docs_path)
    tasks = read_jsonl(tasks_path)
    _validate(faq, docs)

    train_faq = [e for e in faq if e["split"] == "train"]
    eval_faq = [e for e in faq if e["split"] == "eval"]
    train_docs = [d for d in docs if d["split"] == "train"]
    if not train_faq or not eval_faq:
        raise ValueError("train/eval 划分不能为空")

    # 1) 预训练语料：文档摘要 + 同一 FAQ 的多种问法（仅 train 划分）
    blocks: list[str] = []
    for doc in train_docs:
        blocks.append(f"【{doc['title']}】\n{doc['summary']}")
    for entry in train_faq:
        blocks.append(f"【{entry['title']}】\n问：{entry['question']}\n答：{entry['answer']}")
        blocks.append(f"用户：{_strip_question(entry['question'])}？\n助手：{entry['answer']}")
        blocks.append(f"同学问：{_strip_question(entry['question'])}？\n老师答：{entry['answer']}")
        blocks.append(f"{entry['title']}：{entry['answer']}")
        blocks.append(f"关于{entry['title']}：{entry['answer']}")
    corpus = "\n\n".join(blocks) + "\n"
    (out_dir / "corpus.txt").write_text(corpus, encoding="utf-8")

    # 2) 指令微调数据
    sft: list[dict[str, Any]] = []
    for entry in train_faq:
        for question in paraphrase(entry["question"]):
            sft.append({
                "prompt": f"问：{question}\n答：",
                "response": entry["answer"],
                "doc_id": entry["doc_id"],
            })

    # 3) 偏好对比数据：错误答案取其他文档的答案（模拟"答非所问"）
    rng = random.Random(0)
    pref: list[dict[str, Any]] = []
    for entry in train_faq:
        others = [o for o in train_faq if o["doc_id"] != entry["doc_id"]]
        for question in paraphrase(entry["question"])[:2]:
            rejected = rng.choice(others)["answer"]
            pref.append({
                "prompt": f"问：{question}\n答：",
                "chosen": entry["answer"],
                "rejected": rejected,
                "doc_id": entry["doc_id"],
            })

    # 4) RAG 评估集：只使用 eval 划分的 FAQ
    eval_qa: list[dict[str, Any]] = []
    for entry in eval_faq:
        for question in paraphrase(entry["question"]):
            eval_qa.append({
                "query": question,
                "gold_doc_ids": [entry["doc_id"]],
                "gold_keywords": extract_keywords(entry["answer"]),
            })

    write_jsonl(out_dir / "sft.jsonl", sft)
    write_jsonl(out_dir / "pref.jsonl", pref)
    write_jsonl(out_dir / "eval_qa.jsonl", eval_qa)
    write_jsonl(out_dir / "agent_tasks.jsonl", tasks)

    stats = {
        "num_docs": len(docs),
        "num_docs_train": len(train_docs),
        "num_faq": len(faq),
        "num_faq_train": len(train_faq),
        "num_faq_eval": len(eval_faq),
        "num_sft": len(sft),
        "num_pref": len(pref),
        "num_eval_qa": len(eval_qa),
        "num_agent_tasks": len(tasks),
        "corpus_chars": len(corpus),
        "vocab_hint": len({ch for ch in corpus if not ch.isspace()}),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 MiniLLM Studio 训练/评估数据")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="输出目录（默认 data/processed）")
    args = parser.parse_args(argv)
    stats = build(out_dir=args.out)
    print("数据构建完成：")
    for key, value in stats.items():
        print(f"  {key:>16}: {value}")
    print(f"输出目录: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
