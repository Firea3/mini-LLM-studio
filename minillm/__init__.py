"""MiniLLM Studio：轻量级 RLHF + RAG + Agent 全链路实践平台。

本包按分层架构组织：

- ``config``     配置层：dataclass 描述 + YAML 局部覆盖
- ``types``      跨模块的数据契约（Document / SearchHit / AgentTrace ...）
- ``registry``   通用注册表，支撑 Factory / Strategy 模式
- ``tokenizer``  字符级分词器
- ``models``     TinyGPT、LoRA、奖励模型
- ``train``      预训练 / SFT / 奖励建模 / PPO(RLHF) 训练器
- ``rl``         经典强化学习：GridWorld + DQN
- ``rag``        检索增强生成：BM25 检索 + 生成器 + 引用溯源
- ``agent``      ReAct Agent：工具注册表 + 规划器 + 执行循环
- ``serving``    FastAPI 服务与门面层（AppService）
- ``evaluation`` 离线评估（RAG 召回 / Agent 任务成功率）
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
