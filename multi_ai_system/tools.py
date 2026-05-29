"""
multi_ai_system / tools.py
工具系统 —— 为 Agent Worker 提供可调用的工具。

内置工具：
  - web_search    搜索互联网获取信息
  - execute_python 在子进程中执行 Python 代码
  - execute_shell  执行系统命令

扩展方式：
  >>> registry = get_tool_registry()
  >>> registry.register(ToolDef(name="my_tool", ...))
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """一个工具的完整定义"""
    name: str
    description: str
    parameters: dict           # JSON Schema — 参数定义
    handler: Callable[..., Awaitable[str]]   # 异步执行函数
    require_approval: bool = False  # 是否需要用户确认（预留）


# ---------------------------------------------------------------------------
# 工具注册中心
# ---------------------------------------------------------------------------

class ToolRegistry:
    """全局工具注册中心，管理所有可用工具"""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._register_builtins()

    # ── 注册 ──────────────────────────────────────────────────────────

    def register(self, tool: ToolDef) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        """按名称获取工具"""
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """列出所有已注册工具的名称"""
        return list(self._tools.keys())

    # ── OpenAI function-calling 格式 ─────────────────────────────────

    def get_definitions(self, tool_names: list[str] | None = None) -> list[dict]:
        """
        返回 OpenAI / DeepSeek function-calling 格式的工具定义列表。
        tool_names=None 时返回全部。
        """
        names = set(tool_names) if tool_names else set(self._tools.keys())
        result: list[dict] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is None:
                continue
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return result

    # ── 执行 ─────────────────────────────────────────────────────────

    async def execute(self, name: str, arguments: dict) -> str:
        """按名称 + 参数执行工具，返回文本结果"""
        tool = self._tools.get(name)
        if tool is None:
            return f"[错误] 未找到工具: {name}"
        try:
            return await tool.handler(**arguments)
        except TypeError as e:
            return f"[参数错误] {name}: {e}"
        except Exception as e:
            return f"[工具执行异常] {name}: {e}"

    # ── 内置工具实现 ─────────────────────────────────────────────────

    def _register_builtins(self) -> None:
        self.register(ToolDef(
            name="web_search",
            description="搜索互联网获取最新信息。用于查询技术文档、最新动态、事实信息等。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，中英文均可，尽量具体",
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search,
        ))

        self.register(ToolDef(
            name="execute_python",
            description="在子进程中执行 Python 代码并返回输出。用于数学计算、数据处理、代码验证等。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "完整 Python 代码，将写入临时文件后执行",
                    },
                },
                "required": ["code"],
            },
            handler=self._execute_python,
        ))

        self.register(ToolDef(
            name="execute_shell",
            description="执行一条系统命令并返回 stdout/stderr。用于文件操作、版本检查、环境诊断等。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令",
                    },
                },
                "required": ["command"],
            },
            handler=self._execute_shell,
        ))

    # ── 具体实现 ─────────────────────────────────────────────────────

    async def _web_search(self, query: str) -> str:
        """通过 DuckDuckGo Instant Answer API 搜索"""
        try:
            import httpx
        except ImportError:
            return "[错误] httpx 未安装，无法搜索"

        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1,
                        "skip_disambig": 1,
                    },
                    headers={"User-Agent": "MultiAI-Orchestrator/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return f"[搜索请求失败] {e}"

        lines: list[str] = []

        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            lines.append(f"📄 摘要: {abstract}")

        answer = (data.get("Answer") or "").strip()
        if answer:
            lines.append(f"💡 直接答案: {answer}")

        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                lines.append(f"  • {topic['Text']}")

        if not lines:
            return f"未找到「{query}」的相关结果。"

        return "\n".join(lines)

    async def _execute_python(self, code: str) -> str:
        """执行 Python 代码（子进程，30s 超时）"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30,
            )
        except asyncio.TimeoutError:
            return "[超时] Python 代码执行超过 30 秒"
        except FileNotFoundError:
            return "[错误] 未找到 python 命令"

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()

        parts: list[str] = []
        if err:
            parts.append(f"[stderr]\n{err}")
        if out:
            parts.append(f"[stdout]\n{out}")
        return "\n".join(parts) if parts else "(无输出)"

    async def _execute_shell(self, command: str) -> str:
        """执行系统命令（子进程，60s 超时）"""
        try:
            if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
                return "[跳过] 容器环境不允许执行 shell 命令"

            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60,
            )
        except asyncio.TimeoutError:
            return "[超时] 命令执行超过 60 秒"
        except Exception as e:
            return f"[执行失败] {e}"

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()

        parts: list[str] = []
        if err:
            parts.append(f"[stderr]\n{err}")
        if out:
            parts.append(f"[stdout]\n{out}")
        return "\n".join(parts) if parts else "(无输出)"


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册中心（懒加载单例）"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
