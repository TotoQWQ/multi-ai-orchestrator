"""
multi_ai_system / agent.py
Agent 循环 —— 让 Worker 从单次 LLM 调用升级为多步推理 Agent。

核心流程：
  1. 发送系统提示 + 用户任务 + 工具列表给 LLM
  2. 如果 LLM 返回文本 → 视为最终答案，结束
  3. 如果 LLM 返回 tool_calls → 逐个执行，结果追加到对话 → 回到步骤 1
  4. 超过 max_iterations 仍未给出答案 → 强制要求总结

依赖 DeepSeek 原生 function-calling（兼容 OpenAI 格式）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from tools import get_tool_registry, ToolRegistry


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """Agent 循环中的一步"""
    step_num: int
    thinking: str = ""          # 思维链（reasoning_content）
    tool_name: str = ""         # 调用的工具名
    tool_args: dict = field(default_factory=dict)   # 工具参数
    tool_result: str = ""       # 工具执行结果（截断前 2000 字）
    content: str = ""           # LLM 直接输出的文本（无工具调用时）


@dataclass
class AgentResult:
    """Agent 循环的完整执行结果"""
    final_answer: str
    steps: list[AgentStep]
    total_steps: int


# ---------------------------------------------------------------------------
# Agent 循环
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Agent 思考-行动循环。

    用法:
        loop = AgentLoop(client, system_prompt, tools=["web_search", "execute_python"])
        result = await loop.run(task="分析 Python 3.13 的新特性")
        print(result.final_answer)
    """

    def __init__(
        self,
        client,                          # DeepSeekClient 实例
        system_prompt: str,
        tools: list[str] | None = None,
        max_iterations: int = 10,
        on_event: Callable | None = None,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.tool_names = tools or []
        self.max_iterations = max_iterations
        self.on_event = on_event          # (event_type, data) -> None
        self.tool_registry = get_tool_registry()

    # ── 主循环 ──────────────────────────────────────────────────────

    async def run(self, task: str, temperature: float = 0.7) -> AgentResult:
        """运行 Agent 循环，处理一个任务直到 LLM 给出最终答案"""
        steps: list[AgentStep] = []
        tool_defs = self.tool_registry.get_definitions(self.tool_names)

        messages: list[dict] = [
            {"role": "system", "content": self._build_agent_system_prompt()},
            {"role": "user", "content": task},
        ]

        for i in range(self.max_iterations):
            step = AgentStep(step_num=i + 1)

            # —— 调用 LLM ——
            try:
                if tool_defs:
                    response = await self.client.client.chat.completions.create(
                        model=self.client.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=8192,
                        tools=tool_defs,
                    )
                else:
                    # 没有工具时退化为普通调用
                    response = await self.client.client.chat.completions.create(
                        model=self.client.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=8192,
                    )
            except Exception as e:
                step.content = f"[Agent LLM 调用失败] {e}"
                steps.append(step)
                self._emit("agent_error", {"step": i + 1, "error": str(e)})
                break

            choice = response.choices[0]
            msg = choice.message

            # —— 有 tool_calls？ ——
            if msg.tool_calls:
                # 将 assistant 消息（含 tool_calls）追加到对话历史
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                }
                tool_call_blocks: list[dict] = []
                for tc in msg.tool_calls:
                    tool_call_blocks.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
                assistant_msg["tool_calls"] = tool_call_blocks
                messages.append(assistant_msg)

                # 执行每个工具调用
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    step.tool_name = tool_name
                    step.tool_args = tool_args

                    self._emit("tool_call", {
                        "step": i + 1,
                        "tool": tool_name,
                        "args": tool_args,
                    })

                    result_text = await self.tool_registry.execute(tool_name, tool_args)
                    # 工具结果只保留前 2000 字，避免上下文爆炸
                    step.tool_result = result_text[:2000]

                    self._emit("tool_result", {
                        "step": i + 1,
                        "tool": tool_name,
                        "result_preview": result_text[:300],
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text[:2000],
                    })

                steps.append(step)
                continue   # 回到循环顶部，让 LLM 基于工具结果继续思考

            # —— 无 tool_calls → 最终答案 ——
            step.content = msg.content or ""
            steps.append(step)

            self._emit("agent_final", {
                "step": i + 1,
                "content_preview": step.content[:200],
            })

            return AgentResult(
                final_answer=step.content,
                steps=steps,
                total_steps=len(steps),
            )

        # —— 超过最大迭代次数 → 强制总结 ——
        messages.append({
            "role": "user",
            "content": (
                "你已经完成了多轮工具调用。请基于以上所有工具调用结果，"
                "给出最终的综合性答案。不要再调用工具，直接输出最终结论。"
            ),
        })
        try:
            response = await self.client.client.chat.completions.create(
                model=self.client.model,
                messages=messages,
                temperature=temperature,
                max_tokens=8192,
            )
            final = response.choices[0].message.content or ""
        except Exception as e:
            final = f"[Agent 超过 {self.max_iterations} 次迭代，且强制总结失败: {e}]"

        return AgentResult(
            final_answer=final,
            steps=steps,
            total_steps=len(steps),
        )

    # ── 内部辅助 ────────────────────────────────────────────────────

    def _build_agent_system_prompt(self) -> str:
        """在原始 system_prompt 基础上追加 Agent 行为指引"""
        extra = (
            "\n\n## Agent 模式\n"
            "你是一个具备工具调用能力的 Agent。\n"
            "1. 先仔细思考任务，判断是否需要调用工具获取信息或执行操作。\n"
            "2. 如果需要，调用合适的工具（可同时调用多个），等待结果后再继续思考。\n"
            "3. 当信息充分时，给出最终答案，此时不要再调用工具。\n"
            "4. 如果工具调用失败或返回错误，尝试调整参数重试，或基于已有信息给出最佳答案。\n"
            "5. 工具返回的结果可能较长，请提取关键信息整合到最终答案中。"
        )
        return self.system_prompt + extra

    def _emit(self, event_type: str, data: dict) -> None:
        """触发事件回调"""
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass
