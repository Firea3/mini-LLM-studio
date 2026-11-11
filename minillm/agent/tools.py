"""Agent 工具层：安全计算器、知识检索、文本统计、单位换算 + 工具注册表。

设计要点
--------
- Strategy：``BaseTool`` 定义统一 ``run(**kwargs) -> ToolResult`` 接口，各工具是独立策略；
- Factory + Registry：``TOOL_REGISTRY`` 注册工具工厂，``build_tools`` 按配置装配；
- 安全：计算器**绝不使用 eval**，而是 ``ast`` 解析 + 节点白名单求值，从机制上拒绝
  ``import os`` / ``__import__("os").system(...)`` 等任意代码执行。
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

from minillm.registry import Registry
from minillm.types import ToolResult
from minillm.utils.logging import get_logger

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------- 计算器
# 只允许数字、四则运算、括号、小数点、百分号与空白
_ALLOWED_EXPR_RE = re.compile(r"^[0-9+\-*/%().\s]+$")
_MAX_EXPR_LEN = 200
# 从自然语言里抽取算式候选（首尾必须是数字或括号，避免吞掉无关字符）
_EXPR_CANDIDATE_RE = re.compile(r"[0-9(][0-9+\-*/().%\s]*[0-9)]")
_MAX_RESULT = 10**12


def extract_math_expression(text: str) -> str | None:
    """从自然语言任务中抽取最长的合法算式片段（不含 ``=``，如 ``12 * (3 + 4)``）。"""
    if not text:
        return None
    best = ""
    for match in _EXPR_CANDIDATE_RE.finditer(text):
        candidate = match.group(0).strip()
        if not any(op in candidate for op in "+-*/%"):
            continue
        if not any(ch.isdigit() for ch in candidate):
            continue
        if len(candidate) > len(best):
            best = candidate
    return best or None


def _format_number(value: int | float) -> str:
    """结果格式化：整数不带小数点，浮点保留最多 10 位有效数字。"""
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


def _eval_ast(node: ast.AST) -> int | float:
    """对 ast 表达式做白名单递归求值，遇到任何非白名单节点直接报错。"""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("表达式中只允许数字常量")
        return node.value
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("只允许正负号一元运算")
    if isinstance(node, ast.BinOp):
        left, right = _eval_ast(node.left), _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("除数不能为 0")
            result = left / right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("取模的除数不能为 0")
            result = left % right
        else:
            raise ValueError("只允许 + - * / % 运算")
        if abs(result) > _MAX_RESULT:
            raise ValueError("计算结果过大，已拒绝")
        return result
    raise ValueError(f"表达式中包含不允许的语法节点: {type(node).__name__}")


class BaseTool(ABC):
    """工具基类：子类必须定义 ``name`` / ``description`` 并实现 ``run``。"""

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """执行工具；失败时返回 ``ToolResult(ok=False, error=...)``，不向调用方抛异常。"""

    def describe(self) -> dict[str, str]:
        """返回给 LLM / API 展示的工具说明。"""
        return {"name": self.name, "description": self.description}


class CalculatorTool(BaseTool):
    """安全四则运算计算器（ast 白名单求值，禁止任意代码执行）。"""

    name = "calculator"
    description = "计算数学表达式（支持 + - * / % 与括号），参数 expression。"

    def run(self, expression: str = "", text: str = "", **kwargs: Any) -> ToolResult:
        raw = str(expression or text or kwargs.get("query") or "").strip()
        if not raw:
            return ToolResult(ok=False, output="", error="计算器需要非空的 expression 参数")
        if len(raw) > _MAX_EXPR_LEN:
            return ToolResult(ok=False, output="", error=f"表达式过长（> {_MAX_EXPR_LEN} 字符）")
        expr = raw.rstrip("=？? ")
        if not _ALLOWED_EXPR_RE.fullmatch(expr):
            # 仅对含中文的自然语言输入做算式抽取；纯 ASCII 非法输入直接拒绝，
            # 避免 eval('2+2')、__import__('os') 之类的构造被"抽取出算式"而误判为合法。
            if not re.search(r"[\u4e00-\u9fff]", raw):
                return ToolResult(ok=False, output="", error="非法表达式：只允许数字与 + - * / % ( ) .")
            extracted = extract_math_expression(raw)
            if extracted is None:
                return ToolResult(ok=False, output="", error="非法表达式：只允许数字与 + - * / % ( ) .")
            expr = extracted
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval_ast(tree)
        except ZeroDivisionError as exc:
            return ToolResult(ok=False, output="", error=f"除零错误: {exc}")
        except SyntaxError:
            return ToolResult(ok=False, output="", error=f"非法表达式: {raw!r}")
        except ValueError as exc:
            return ToolResult(ok=False, output="", error=f"非法表达式: {exc}")
        except (TypeError, OverflowError) as exc:
            return ToolResult(ok=False, output="", error=f"计算失败: {exc}")
        return ToolResult(ok=True, output=_format_number(value))


class KnowledgeSearchTool(BaseTool):
    """知识库检索工具：返回 top_k 片段与来源编号（依赖 RAGPipeline）。"""

    name = "knowledge_search"
    description = "检索本地知识库/FAQ，返回带 [编号] 的片段与来源，参数 query。"

    def __init__(self, pipeline: Any, top_k: int = 3) -> None:
        if pipeline is None:
            raise ValueError("KnowledgeSearchTool 需要 RAGPipeline 实例")
        self.pipeline = pipeline
        self.top_k = int(top_k)

    def run(self, query: str = "", top_k: int | None = None, **kwargs: Any) -> ToolResult:
        query = str(query or kwargs.get("question") or kwargs.get("text") or "").strip()
        if not query:
            return ToolResult(ok=False, output="", error="知识检索需要非空的 query 参数")
        k = int(top_k or self.top_k)
        try:
            hits = self.pipeline.search(query, top_k=k)
        except Exception as exc:  # 检索器异常也要转成 observation 文本
            return ToolResult(ok=False, output="", error=f"知识检索失败: {exc}")
        if not hits:
            return ToolResult(ok=False, output="", error="未检索到相关内容")
        lines = [
            f"[{idx}] {hit.doc.doc_id} / {hit.doc.title}（score={hit.score:.3f}）：{hit.snippet}"
            for idx, hit in enumerate(hits, start=1)
        ]
        return ToolResult(ok=True, output="\n".join(lines))


class TextStatsTool(BaseTool):
    """文本统计：字符数 / 词数 / 行数。

    ``词数`` 定义：英文/数字按空格词计，中文按单字计（中文单字 ≈ 一个"字"），便于中文场景理解。
    """

    name = "text_stats"
    description = "统计文本的字符数、词数、行数，参数 text。"
    _WORD_RE = re.compile(r"[A-Za-z0-9_]+")
    _CJK_RE = re.compile(r"[\u4e00-\u9fff]")

    def run(self, text: str = "", **kwargs: Any) -> ToolResult:
        text = str(text if text is not None else kwargs.get("query") or "")
        if not text:
            return ToolResult(ok=False, output="", error="文本统计需要非空的 text 参数")
        chars = len(text)
        words = len(self._WORD_RE.findall(text)) + len(self._CJK_RE.findall(text))
        lines = text.count("\n") + 1
        return ToolResult(ok=True, output=f"字符数: {chars}，词数: {words}，行数: {lines}")


# ---------------------------------------------------------------------- 单位换算
# 别名 -> 规范单位 key（同类别才能互转）
UNIT_ALIASES: dict[str, str] = {
    # 温度
    "摄氏度": "c", "℃": "c", "°c": "c", "celsius": "c", "c": "c",
    "华氏度": "f", "℉": "f", "°f": "f", "fahrenheit": "f", "f": "f",
    # 长度（基准：米）
    "千米": "km", "公里": "km", "km": "km", "kilometer": "km",
    "米": "m", "公尺": "m", "m": "m", "meter": "m",
    "厘米": "cm", "公分": "cm", "cm": "cm",
    "毫米": "mm", "mm": "mm",
    # 质量（基准：克）
    "千克": "kg", "公斤": "kg", "kg": "kg",
    "克": "g", "g": "g",
    # 时间（基准：秒）
    "小时": "h", "h": "h", "hour": "h",
    "分钟": "min", "min": "min", "minute": "min",
    "秒": "s", "s": "s", "second": "s",
}
UNIT_DISPLAY: dict[str, str] = {
    "c": "摄氏度", "f": "华氏度",
    "m": "米", "km": "千米", "cm": "厘米", "mm": "毫米",
    "g": "克", "kg": "千克",
    "s": "秒", "min": "分钟", "h": "小时",
}
UNIT_CATEGORY: dict[str, str] = {
    "c": "temperature", "f": "temperature",
    "m": "length", "km": "length", "cm": "length", "mm": "length",
    "g": "mass", "kg": "mass",
    "s": "time", "min": "time", "h": "time",
}
# 同类别的换算系数（换算到基准单位）
UNIT_FACTOR: dict[str, float] = {
    "m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
    "g": 1.0, "kg": 1000.0,
    "s": 1.0, "min": 60.0, "h": 3600.0,
}
_UNIT_ALIAS_PATTERN = re.compile(
    "|".join(sorted((re.escape(alias) for alias in UNIT_ALIASES), key=len, reverse=True)),
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

SUPPORTED_UNITS = "摄氏度/华氏度、米/千米/厘米/毫米、克/千克、秒/分钟/小时"


def normalize_unit(unit: str) -> str | None:
    """单位名 -> 规范 key；未知单位返回 ``None``。"""
    key = str(unit or "").strip().lower().strip("度")
    if not key:
        return None
    if key in UNIT_ALIASES:
        return UNIT_ALIASES[key]
    # 兼容用户输入带"度"后缀（如 "摄氏"）
    key2 = key + "度"
    return UNIT_ALIASES.get(key2)


def find_supported_units(text: str) -> list[str]:
    """按出现顺序返回文本中受支持的单位规范 key（用于规划器做意图识别）。"""
    if not text:
        return []
    seen: list[str] = []
    for match in _UNIT_ALIAS_PATTERN.finditer(text):
        key = UNIT_ALIASES.get(match.group(0).lower())
        if key and key not in seen:
            seen.append(key)
    return seen


def parse_conversion_text(text: str) -> tuple[float, str, str]:
    """从自然语言里解析 ``(数值, 源单位, 目标单位)``；失败抛 ``ValueError``。"""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("单位换算需要非空文本，例如：100 摄氏度等于多少华氏度")
    # 依次定位"数字 + 单位"与紧随其后的目标单位
    number_matches = list(_NUMBER_RE.finditer(raw))
    unit_matches = list(_UNIT_ALIAS_PATTERN.finditer(raw))
    if not unit_matches:
        raise ValueError(f"未识别到受支持的单位（支持: {SUPPORTED_UNITS}）")
    from_match = None  # (number, unit) 源
    to_unit: str | None = None
    for number in number_matches:
        following = [m for m in unit_matches if m.start() >= number.end()]
        if not following:
            continue
        from_match = (float(number.group(0)), UNIT_ALIASES[following[0].group(0).lower()])
        later = [m for m in unit_matches if m.start() >= following[0].end()]
        if later:
            to_unit = UNIT_ALIASES[later[0].group(0).lower()]
        break
    if from_match is None:
        # 没有"数字+单位"时，退化为"源单位…目标单位"（value 由调用方给出）
        if len(unit_matches) >= 2:
            return 0.0, UNIT_ALIASES[unit_matches[0].group(0).lower()], UNIT_ALIASES[unit_matches[1].group(0).lower()]
        raise ValueError("未识别到'数值 + 源单位'，例如：36 摄氏度等于多少华氏度")
    value, from_unit = from_match
    if to_unit is None:
        if len(unit_matches) >= 2:
            to_unit = UNIT_ALIASES[unit_matches[1].group(0).lower()]
        else:
            raise ValueError("未识别到目标单位，例如：36 摄氏度等于多少华氏度")
    return value, from_unit, to_unit


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """按类别换算（温度用公式，其余用基准单位系数）；跨类别抛 ``ValueError``。"""
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise ValueError("单位必须是字符串")
    from_key, to_key = normalize_unit(from_unit), normalize_unit(to_unit)
    if from_key is None or to_key is None:
        raise ValueError(f"未知单位（支持: {SUPPORTED_UNITS}）")
    category = UNIT_CATEGORY[from_key]
    if category != UNIT_CATEGORY[to_key]:
        raise ValueError(
            f"不支持跨类别换算: {UNIT_DISPLAY[from_key]} -> {UNIT_DISPLAY[to_key]}"
        )
    if from_key == to_key:
        return float(value)
    if category == "temperature":  # 摄氏 <-> 华氏
        if from_key == "c":
            return float(value) * 9.0 / 5.0 + 32.0
        return (float(value) - 32.0) * 5.0 / 9.0
    return float(value) * UNIT_FACTOR[from_key] / UNIT_FACTOR[to_key]


class UnitConvertTool(BaseTool):
    """单位换算：温度、长度、质量、时间；未知/跨类别单位返回错误。"""

    name = "unit_convert"
    description = f"单位换算（支持 {SUPPORTED_UNITS}），可传 text 自然语言或 value/from_unit/to_unit。"

    def run(
        self,
        value: float | str | None = None,
        from_unit: str | None = None,
        to_unit: str | None = None,
        text: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        raw_text = str(text or kwargs.get("query") or "").strip()
        try:
            if value is not None and from_unit and to_unit:
                amount, src, dst = float(value), from_unit, to_unit
            elif raw_text:
                amount, src, dst = parse_conversion_text(raw_text)
            elif value is not None and from_unit and not to_unit:
                return ToolResult(ok=False, output="", error="缺少目标单位 to_unit")
            else:
                return ToolResult(
                    ok=False,
                    output="",
                    error=f"单位换算需要 text（自然语言）或 value/from_unit/to_unit，支持: {SUPPORTED_UNITS}",
                )
            result = convert_units(amount, src, dst)
        except (ValueError, TypeError) as exc:
            return ToolResult(ok=False, output="", error=f"单位换算失败: {exc}")
        src_key, dst_key = normalize_unit(src), normalize_unit(dst)
        assert src_key is not None and dst_key is not None  # convert_units 已校验
        return ToolResult(
            ok=True,
            output=f"{_format_number(amount)}{UNIT_DISPLAY[src_key]} = {_format_number(result)}{UNIT_DISPLAY[dst_key]}",
        )


# ---------------------------------------------------------------------- 工具注册表
class ToolRegistry:
    """工具注册表（Factory + Registry 的运行期容器）。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> BaseTool:
        """注册工具实例；重名抛 ``ValueError``。"""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"只能注册 BaseTool 实例，收到 {type(tool).__name__}")
        if not tool.name:
            raise ValueError("工具必须定义非空的 name")
        if tool.name in self._tools:
            raise ValueError(f"重复注册工具: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> BaseTool:
        """按名称取工具；不存在时抛 ``KeyError``（附可选名称）。"""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"未注册的工具 {name!r}，可选: {self.names()}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def describe(self) -> list[dict[str, str]]:
        """返回所有工具的名称与说明（供 LLM prompt / API 展示）。"""
        return [self._tools[name].describe() for name in self.names()]

    def execute(self, name: str, args: dict[str, Any] | None = None) -> ToolResult:
        """执行工具：参数错误/未知工具/工具内部异常都收敛为 ``ToolResult(ok=False)``。"""
        if not isinstance(args, dict) and args is not None:
            return ToolResult(ok=False, output="", error=f"工具参数必须是 dict，收到 {type(args).__name__}")
        try:
            tool = self.get(name)
        except KeyError as exc:
            return ToolResult(ok=False, output="", error=str(exc))
        try:
            return tool.run(**(args or {}))
        except TypeError as exc:  # 参数名不匹配
            return ToolResult(ok=False, output="", error=f"工具 {name} 参数错误: {exc}")
        except Exception as exc:  # 兜底：工具异常不能炸掉 Agent 循环
            LOGGER.warning("工具 %s 执行异常: %s", name, exc)
            return ToolResult(ok=False, output="", error=f"工具 {name} 执行异常: {exc}")


TOOL_FACTORY: Registry[Callable[..., BaseTool]] = Registry("tool")


@TOOL_FACTORY.register("calculator")
def _create_calculator(**_: Any) -> BaseTool:
    """工厂：安全计算器。"""
    return CalculatorTool()


@TOOL_FACTORY.register("text_stats")
def _create_text_stats(**_: Any) -> BaseTool:
    """工厂：文本统计。"""
    return TextStatsTool()


@TOOL_FACTORY.register("unit_convert")
def _create_unit_convert(**_: Any) -> BaseTool:
    """工厂：单位换算。"""
    return UnitConvertTool()


@TOOL_FACTORY.register("knowledge_search")
def _create_knowledge_search(rag_pipeline: Any = None, top_k: int = 3, **_: Any) -> BaseTool:
    """工厂：知识检索（需要 RAGPipeline）。"""
    if rag_pipeline is None:
        raise ValueError("knowledge_search 工具需要 RAGPipeline")
    return KnowledgeSearchTool(rag_pipeline, top_k=top_k)


def build_tools(cfg: Any, rag_pipeline: Any = None) -> ToolRegistry:
    """按 ``cfg.agent.tools`` 装配工具注册表；缺 RAG 时跳过知识检索并告警。"""
    registry = ToolRegistry()
    for name in cfg.agent.tools:
        if name == "knowledge_search" and rag_pipeline is None:
            LOGGER.warning("未提供 RAGPipeline，跳过 knowledge_search 工具")
            continue
        if name not in TOOL_FACTORY:
            LOGGER.warning("未知工具配置项 %r，已跳过（可选: %s）", name, TOOL_FACTORY.names())
            continue
        try:
            registry.register(TOOL_FACTORY.create(name, rag_pipeline=rag_pipeline, top_k=cfg.rag.top_k))
        except ValueError as exc:
            LOGGER.warning("工具 %s 装配失败: %s", name, exc)
    return registry


__all__ = [
    "BaseTool",
    "CalculatorTool",
    "KnowledgeSearchTool",
    "SUPPORTED_UNITS",
    "TOOL_FACTORY",
    "TextStatsTool",
    "ToolRegistry",
    "UnitConvertTool",
    "build_tools",
    "convert_units",
    "extract_math_expression",
    "find_supported_units",
    "normalize_unit",
    "parse_conversion_text",
]
