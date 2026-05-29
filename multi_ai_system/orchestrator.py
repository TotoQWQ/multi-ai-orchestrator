"""
multi_ai_system / orchestrator.py
多AI协作编排系统核心逻辑。

架构：
  Orchestrator（总规划AI）
    ├── 阶段 0：并行执行独立子任务
    ├── 阶段 1：串行执行，可引用阶段 0 的输出
    ├── 阶段 2：串行执行，可引用前序所有输出
    └── … 直到任务完成
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional
import httpx

# Agent 功能（可选依赖，不影响原有逻辑）
try:
    from agent import AgentLoop, AgentResult
    from tools import get_tool_registry
    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False

# 工作流规划器
try:
    from planner import WorkflowPlanner, Workflow, WorkflowStep, WorkflowEvaluation, PlanRoundResult
    _PLANNER_AVAILABLE = True
except ImportError:
    _PLANNER_AVAILABLE = False

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """专职AI助手的配置"""
    name: str
    description: str          # 简短描述，给 Orchestrator 用来分配任务
    system_prompt: str        # 系统提示词，定义该助手的专长和行为
    temperature: float = 0.7
    agent_mode: bool = False  # 是否启用 Agent 模式（支持工具调用）
    tools: list[str] = field(default_factory=list)   # Agent 模式下可用的工具列表
    max_iterations: int = 10  # Agent 最大迭代次数


@dataclass
class SubTask:
    """一个子任务"""
    worker: str               # 分配给哪个助手（name）
    description: str          # 任务描述
    phase: int = 0            # 执行阶段：同 phase 并行，不同 phase 串行
    depends_on: list[str] | None = None  # 依赖的 worker/step_id 列表
    step_id: str = ""         # 工作流步骤 ID（同一 Worker 可多次出现，用 step_id 区分）


@dataclass
class DecompositionResult:
    """任务分解结果"""
    analysis: str
    subtasks: list[SubTask]


@dataclass
class RoundResult:
    """一轮的执行结果"""
    round_num: int
    subtask_results: dict[str, str]          # worker_name -> 输出文本
    subtask_details: list[tuple[str, str, int]]   # (worker_name, description, phase)
    phase_count: int = 1                     # 本轮有多少个阶段
    agent_steps: dict[str, list[dict]] = field(default_factory=dict)  # worker_name -> agent 步骤列表


@dataclass
class SummaryResult:
    """归纳总结结果"""
    summary: str
    key_findings: list[str]
    inconsistencies: list[str]
    is_final: bool
    final_answer: str = ""
    next_round_focus: str = ""


# ---------------------------------------------------------------------------
# DeepSeek API 客户端（异步，兼容 OpenAI 格式）
# ---------------------------------------------------------------------------

class DeepSeekClient:
    """封装 DeepSeek Flash API 的异步调用（支持动态换 key）"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: int = 120,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self.model = model
        self._timeout = timeout
        self._client = None
        self._rebuild()

    def _rebuild(self):
        from openai import AsyncOpenAI
        if not self._api_key:
            self._client = None
            return
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)

    def set_api_key(self, api_key: str):
        self._api_key = api_key
        self._rebuild()

    @property
    def client(self):
        if not self._client:
            raise RuntimeError("API Key 未设置，请在页面上添加 API Key")
        return self._client

    @client.setter
    def client(self, val):
        self._client = val

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        on_token=None,
    ) -> str:
        """调用 DeepSeek Flash 模型（流式）。
           通过 on_token(reasoning_chunk, content_chunk, full_text) 实时推送。"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            full_content = ""
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or ""
                content = delta.content or ""

                if content:
                    full_content += content
                if on_token:
                    on_token(reasoning, content, full_content)

            return full_content
        except Exception as e:
            raise RuntimeError(f"DeepSeek API 调用失败：{e}")


# ---------------------------------------------------------------------------
# JSON 解析工具
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """从 LLM 回复中提取并解析 JSON 对象"""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if start == -1:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start != -1:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1

    raise ValueError(
        f"无法从 LLM 回复中解析出合法 JSON。\n"
        f"回复摘要：{text[:300]}"
    )


# ---------------------------------------------------------------------------
# 多AI编排器核心
# ---------------------------------------------------------------------------

class MultiAIOrchestrator:
    """多AI协作编排器：任务分解 → 分阶段协作执行 → 归纳总结 → 多轮迭代"""

    def __init__(
        self,
        workers: list[WorkerConfig],
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        orchestrator_system_prompt: str | None = None,
        max_concurrent: int = 10,
    ):
        self.workers = workers
        self.workers_map = {w.name: w for w in workers}
        self.client = DeepSeekClient(api_key, base_url, model)
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._on_event = None

        self.orchestrator_system_prompt = (
            orchestrator_system_prompt or self._build_orchestrator_prompt()
        )

    # ── system prompt 构建 ──────────────────────────────────────────────

    def _build_orchestrator_prompt(self) -> str:
        worker_lines = "\n".join(
            f"  - {w.name}：{w.description}" for w in self.workers
        )
        return f"""你是一个多AI协作系统的总规划师（Orchestrator），使用 DeepSeek Flash 模型。

## 职责
1. **分析**用户的大任务
2. **拆分**为多个子任务并按阶段（phase）组织协作流程
3. **综合**各助手输出，得到高质量最终答案
4. **判断**任务是否完成或需要继续

## 可用的专职AI助手
{worker_lines}

## 协作机制
你可以通过 phase（阶段）号来组织 worker 之间的协作：
- **phase 0**：基础任务，互不依赖，并行执行
- **phase 1**：依赖 phase 0 的输出，看到前序结果后执行
- **phase 2+**：依赖前序所有输出，继续深入
- 同 phase 的任务并行执行，不同 phase 串行

合理利用 phase 让 worker 互相配合。例如：
- phase 0：架构师设计系统 → phase 1：审查员审查架构师的方案
- phase 0：数据分析 + 市场调研（并行）→ phase 1：基于两者写综合报告

## 输出原则
- 精炼：用最少的字数说清要点
- 鲜明：用结构化 Markdown 组织内容
- 准确：助手名称严格匹配列表

## 任务分解阶段输出格式
{{
  "analysis": "精炼分析，说明协作策略（各 phase 安排）",
  "subtasks": [
    {{"worker": "助手名称", "description": "子任务描述", "phase": 0}},
    {{"worker": "助手名称", "description": "依赖架构师输出的审查任务", "phase": 1, "depends_on": ["架构设计师"]}},
    {{"worker": "助手名称", "description": "并行任务", "phase": 1}}
  ]
}}

depends_on 用法：
- 可选字段，不填则看到前序所有输出
- 填了则只看到指定助手的输出，用于精准协作
- 仅在 phase>0 时有效

## 归纳总结阶段输出格式
{{
  "summary": "精炼总结",
  "key_findings": ["关键发现"],
  "inconsistencies": [],
  "is_final": true,
  "final_answer": "完整最终答案（Markdown 格式，结构鲜明）",
  "next_round_focus": ""
}}

final_answer 请用 Markdown 组织：标题 ##、列表 -、代码块 ```、加粗 **等。"""

    def set_api_key(self, api_key: str):
        """运行时切换 API Key"""
        self.client.set_api_key(api_key)

    # ── 内部 LLM 调用 ──────────────────────────────────────────────────

    async def _call_llm(self, messages: list[dict], temperature: float = 0.7) -> str:
        if self._on_event:
            def on_token(rea, txt, full):
                if rea:
                    self._on_event("reasoning", {"text": rea, "full": full})
                if txt:
                    self._on_event("token", {"text": txt, "full": full})
            return await self.client.chat(messages, temperature, on_token=on_token)
        return await self.client.chat(messages, temperature)

    # ── 任务分解 ───────────────────────────────────────────────────────

    async def decompose_task(
        self,
        task: str,
        history: str = "",
    ) -> DecompositionResult:
        """将大任务分解为子任务"""
        prefix = "== Decompose ==\n分析并拆解以下任务：\n\n"
        user_msg = prefix + task

        if history:
            user_msg += f"\n\n[History]\n{history}\n\n请基于历史进展调整本轮拆解。"
        else:
            user_msg += "\n\n按 phase 组织多阶段协作。输出 JSON 格式的分解方案。"

        messages = [
            {"role": "system", "content": self.orchestrator_system_prompt},
            {"role": "user", "content": user_msg},
        ]

        response = await self._call_llm(messages)
        data = _extract_json(response)

        subtasks = []
        for st in data.get("subtasks", []):
            deps = st.get("depends_on")
            if isinstance(deps, list):
                deps = [d.strip() for d in deps if isinstance(d, str)]
            else:
                deps = None
            subtasks.append(SubTask(
                worker=st["worker"],
                description=st["description"],
                phase=int(st.get("phase", 0)),
                depends_on=deps or None,
            ))

        return DecompositionResult(
            analysis=data.get("analysis", ""),
            subtasks=subtasks,
        )

    # ── 分阶段协作执行 ─────────────────────────────────────────────────

    async def _fetch_url(self, url: str) -> str:
        """抓取网页内容，返回纯文本摘要（最多5000字）"""
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as cli:
                resp = await cli.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                html = resp.text[:50000]
                # 简单清理：移除 script/style 标签
                html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL|re.IGNORECASE)
                html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL|re.IGNORECASE)
                html = re.sub(r'<[^>]+>', ' ', html)
                html = re.sub(r'\s+', ' ', html).strip()
                return html[:5000] or "(空白页面)"
        except Exception as e:
            return f"[抓取失败] {e}"

    async def _execute_one(self, subtask: SubTask, context: str = "") -> tuple[str, str, list[dict]]:
        """
        执行单个子任务，可附带前序阶段的上下文。自动抓取描述中的 URL。
        返回 (worker_name, result_text, agent_steps)
        """
        async with self._semaphore:
            worker = self.workers_map.get(subtask.worker)
            if not worker:
                return (
                    subtask.worker,
                    f"[错误] 找不到名为 '{subtask.worker}' 的助手。可用：{list(self.workers_map.keys())}",
                    [],
                )

            if self._on_event:
                self._on_event("worker_start", {"worker": subtask.worker, "description": subtask.description[:120], "phase": subtask.phase, "agent_mode": worker.agent_mode})

            full_prompt = subtask.description

            # 自动抓取描述中的 URL
            urls = re.findall(r'https?://[^\s,，。、]+', full_prompt)
            fetched = []
            for url in urls[:2]:  # 最多抓取 2 个
                if self._on_event:
                    self._on_event("worker_start", {"worker": subtask.worker, "description": f"正在抓取… {url[:50]}", "phase": subtask.phase})
                content = await self._fetch_url(url)
                fetched.append(f"网页 [{url}] 的内容：\n{content}")

            if context:
                full_prompt += f"\n\n[前序协作上下文 — 其他助手的输出，供你参考]\n{context}"
            if fetched:
                full_prompt += f"\n\n[自动抓取的网页内容 — 供你分析参考]\n" + "\n\n---\n\n".join(fetched)

            # ── Agent 模式 ──
            if worker.agent_mode and _AGENT_AVAILABLE:
                agent_steps: list[dict] = []

                def _on_agent_event(event_type: str, data: dict):
                    """捕获 Agent 内部事件，转发给 Orchestrator 的 SSE"""
                    if self._on_event:
                        self._on_event("agent_" + event_type, dict(data, worker=subtask.worker))
                    if event_type in ("tool_call", "tool_result", "agent_final"):
                        agent_steps.append(dict(data, type=event_type))

                loop = AgentLoop(
                    client=self.client,
                    system_prompt=worker.system_prompt,
                    tools=worker.tools if worker.tools else None,
                    max_iterations=worker.max_iterations,
                    on_event=_on_agent_event,
                )
                agent_result = await loop.run(
                    task=f"== Execute ==\n{full_prompt}",
                    temperature=worker.temperature,
                )

                # 将 agent steps 转为可序列化的格式
                serializable_steps = []
                for s in agent_result.steps:
                    serializable_steps.append({
                        "step_num": s.step_num,
                        "tool_name": s.tool_name,
                        "tool_args": s.tool_args,
                        "tool_result": s.tool_result[:500],
                        "content": s.content[:500],
                    })

                if self._on_event:
                    self._on_event("worker_done", {
                        "worker": subtask.worker,
                        "phase": subtask.phase,
                        "summary": agent_result.final_answer[:200],
                        "agent_steps_count": len(agent_result.steps),
                    })
                return subtask.worker, agent_result.final_answer, serializable_steps

            # ── 普通模式（原有逻辑） ──
            messages = [
                {"role": "system", "content": worker.system_prompt},
                {"role": "user", "content": f"== Execute ==\n{full_prompt}"},
            ]

            result = await self._call_llm(messages, worker.temperature)

            if self._on_event:
                self._on_event("worker_done", {"worker": subtask.worker, "phase": subtask.phase, "summary": result[:200]})
            return subtask.worker, result, []

    async def execute_phased(self, subtasks: list[SubTask]) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """
        分阶段协作执行：
        - 先按 phase 分组
        - 同 phase 并行执行
        - 不同 phase 串行执行，后 phase 看到前序所有输出
        返回 (all_results, all_agent_steps)
        """
        # 按 phase 分组
        phases: dict[int, list[SubTask]] = {}
        for st in subtasks:
            phases.setdefault(st.phase, []).append(st)

        all_results: dict[str, str] = {}
        all_agent_steps: dict[str, list[dict]] = {}
        sorted_phases = sorted(phases.keys())

        if self._on_event:
            self._on_event("phase", {"phase": "execute", "status": "start", "phase_count": len(sorted_phases)})

        for phase_i, phase_num in enumerate(sorted_phases):
            phase_tasks = phases[phase_num]

            if self._on_event:
                self._on_event("execute_phase", {"phase_num": phase_num, "phase_index": phase_i, "total": len(sorted_phases), "workers": [st.worker for st in phase_tasks]})

            # 构建上下文：支持选择性依赖（depends_on）或全部前序输出
            context = ""
            if all_results:
                # 收集所有需要传递的 worker 输出
                needed = set()
                for st in phase_tasks:
                    if st.depends_on:
                        needed.update(st.depends_on)
                    else:
                        needed.update(all_results.keys())  # 全部
                if needed:
                    context_lines = []
                    for name, content in all_results.items():
                        if name not in needed:
                            continue
                        truncated = content[:800] if len(content) > 800 else content
                        context_lines.append(f"[{name} 的输出]\n{truncated}")
                    context = "\n\n---\n\n".join(context_lines) if context_lines else ""

            # 执行当前 phase 的所有任务（并行）
            tasks = [self._execute_one(st, context) for st in phase_tasks]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for i, res in enumerate(results_list):
                st = phase_tasks[i]
                key = st.step_id if st.step_id else st.worker
                if isinstance(res, Exception):
                    all_results[key] = f"[执行异常] {res}"
                else:
                    worker_name, content, agent_steps = res
                    all_results[key] = content
                    if agent_steps:
                        all_agent_steps[key] = agent_steps

        if self._on_event:
            self._on_event("phase", {"phase": "execute", "status": "done"})

        return all_results, all_agent_steps

    # ── 归纳总结 ───────────────────────────────────────────────────────

    async def summarize_round(
        self,
        original_task: str,
        round_result: RoundResult,
    ) -> SummaryResult:
        """对一轮的执行结果进行归纳总结"""
        prefix = "== Summarize ==\n综合分析以下执行结果：\n\n"
        parts = [prefix + f"原始任务：{original_task}\n"]
        parts.append(f"===== 第 {round_result.round_num} 轮执行结果 =====\n")

        for worker_name, description, phase in round_result.subtask_details:
            content = round_result.subtask_results.get(worker_name, "[无结果]")
            if len(content) > 1500:
                content = content[:1500] + f"\n\n[... 以下截断，共{len(content)}字符]"
            parts.append(f"\n### [阶段{phase}][{worker_name}] 子任务：{description}\n\n{content}\n")

        user_msg = "".join(parts)
        messages = [
            {"role": "system", "content": self.orchestrator_system_prompt},
            {"role": "user", "content": user_msg},
        ]

        response = await self._call_llm(messages)
        data = _extract_json(response)

        return SummaryResult(
            summary=data.get("summary", ""),
            key_findings=data.get("key_findings", []),
            inconsistencies=data.get("inconsistencies", []),
            is_final=data.get("is_final", True),
            final_answer=data.get("final_answer", ""),
            next_round_focus=data.get("next_round_focus", ""),
        )

    # ── 工作流转换 ──────────────────────────────────────────────────

    def _workflow_to_subtasks(self, workflow: Workflow) -> list[SubTask]:
        """将 Plan 生成的 Workflow 转换为 SubTask 列表（兼容现有执行引擎）"""
        subtasks = []
        for phase_num in sorted(workflow.phases.keys()):
            for step in workflow.phases[phase_num]:
                # depends_on 中的 step_id 需确保在前序阶段
                subtasks.append(SubTask(
                    worker=step.worker,
                    description=f"[预期产出: {step.expected_output}]\n\n{step.description}",
                    phase=step.phase,
                    depends_on=step.depends_on,
                    step_id=step.step_id,
                ))
        return subtasks

    @staticmethod
    def _format_results_for_replan(
        workflow: Workflow,
        step_results: dict[str, str],
    ) -> str:
        """将执行结果格式化为 Planner 可读的文本，用于重新规划"""
        parts = [f"工作流分析：{workflow.analysis}\n"]
        for phase_num in sorted(workflow.phases.keys()):
            for step in workflow.phases[phase_num]:
                content = step_results.get(step.step_id, "[未执行]")
                preview = content[:800] if len(content) > 800 else content
                parts.append(
                    f"## Phase {step.phase} [{step.worker}] {step.step_id}\n"
                    f"预期产出：{step.expected_output}\n"
                    f"实际产出：{preview}\n"
                )
        return "\n".join(parts)

    # ── 主编排流程（规划优先模式） ──────────────────────────────────

    async def run(
        self,
        task: str,
        max_rounds: int = 3,
        verbose: bool = True,
        on_round_complete=None,
        on_event=None,
    ) -> str:
        """
        主编排流程（规划优先模式）：
          1. Planner AI 分析任务，生成完整工作流
          2. Workers 按工作流分阶段执行
          3. Evaluator AI 评估执行质量
          4. 不合格 → Planner 修正工作流 → 回到步骤 2（最多 max_rounds 次）
          5. 合格 → 输出最终答案

        与旧版「每轮重新分解」的区别：
          - 旧：每轮从零分解 → 执行 → 总结 → 重复
          - 新：一次规划 → 执行 → 评估 → 仅不合格时修正规划
        """
        self._on_event = on_event

        if not _PLANNER_AVAILABLE:
            return await self._run_legacy(task, max_rounds, verbose, on_round_complete, on_event)

        planner = WorkflowPlanner(self.client, self.workers)
        planner.set_event_callback(on_event)

        all_iterations: list[PlanRoundResult] = []
        previous_results_text = ""
        previous_issues = ""

        for iteration in range(1, max_rounds + 1):
            if self._on_event:
                self._on_event("round_start", {"round": iteration, "mode": "plan_first"})
            if verbose:
                self._print_separator()
                self._print(f"第 {iteration} 轮：规划工作流", bold=True)

            # ── 1. 规划工作流 ──
            if self._on_event:
                self._on_event("phase", {"phase": "plan", "status": "start", "round": iteration})
            if verbose:
                self._print("Planner 正在设计工作流…")

            try:
                workflow = await planner.plan(task, previous_results_text, previous_issues)
            except Exception as e:
                if self._on_event:
                    self._on_event("error", {"message": f"工作流规划失败: {e}"})
                self._print(f"工作流规划失败：{e}", color="red")
                break

            if verbose:
                self._print(f"规划分析：{workflow.analysis}")
                self._print(f"工作流：{len(workflow.phases)} 个阶段，{workflow.total_steps} 个步骤")
                for phase_num in sorted(workflow.phases.keys()):
                    steps = workflow.phases[phase_num]
                    self._print(f"  Phase {phase_num}（{'并行' if len(steps) > 1 else '单任务'}）：")
                    for st in steps:
                        self._print(f"    ├─ [{st.step_id}] {st.worker}: {st.description[:80]}…")

            # ── 2. 执行工作流 ──
            subtasks = self._workflow_to_subtasks(workflow)
            if not subtasks:
                if verbose:
                    self._print("工作流没有步骤，结束", color="yellow")
                break

            if verbose:
                self._print(f"\n开始按工作流执行…")
            results, agent_steps_map = await self.execute_phased(subtasks)

            if verbose:
                self._print("所有步骤执行完成：")
                for step_id, content in results.items():
                    preview = content[:120].replace("\n", " ")
                    self._print(f"  [{step_id}] {preview}…")

            # ── 3. 评估质量 ──
            if verbose:
                self._print(f"\nEvaluator 正在评估执行质量…")
            evaluation = await planner.evaluate(task, workflow, results)

            if verbose:
                self._print(f"质量评分：{evaluation.score:.1f}/10 {'✓ 通过' if evaluation.is_satisfactory else '✗ 需修正'}")
                if evaluation.issues:
                    for issue in evaluation.issues:
                        self._print(f"  ⚠ {issue}")

            # ── 4. 记录 ──
            plan_result = PlanRoundResult(
                iteration=iteration,
                workflow=workflow,
                step_results=results,
                agent_steps=agent_steps_map,
                evaluation=evaluation,
            )
            all_iterations.append(plan_result)

            if on_round_complete:
                on_round_complete(iteration, workflow, results, evaluation)

            # ── 5. 判断 ──
            if evaluation.is_satisfactory:
                if verbose:
                    self._print(f"\n任务完成！质量评分 {evaluation.score:.1f}/10", bold=True)
                final = evaluation.final_answer or self._build_fallback_answer(workflow, results)
                if self._on_event:
                    self._on_event("done", {"result": final, "score": evaluation.score})
                return final

            # ── 6. 准备下一轮修正 ──
            previous_results_text = self._format_results_for_replan(workflow, results)
            previous_issues = evaluation.suggestions

            if verbose and iteration < max_rounds:
                self._print(f"\n评估未通过，Planner 将基于反馈修正工作流…")
                self._print(f"改进方向：{evaluation.suggestions[:200]}…")

        # ── 所有迭代结束 ──
        if not all_iterations:
            msg = "未能完成任何迭代。"
            if self._on_event:
                self._on_event("done", {"result": msg})
            return msg

        # 取最后一次评估的 final_answer，若无则综合
        last_eval = all_iterations[-1].evaluation
        if last_eval.final_answer:
            if self._on_event:
                self._on_event("done", {"result": last_eval.final_answer})
            return last_eval.final_answer

        # 超过迭代次数：综合所有结果
        if verbose:
            self._print("\n超过最大迭代次数，综合所有结果…")
        if self._on_event:
            self._on_event("phase", {"phase": "final_synthesis", "status": "start"})

        last_wf = all_iterations[-1].workflow
        last_results = all_iterations[-1].step_results
        fallback = self._build_fallback_answer(last_wf, last_results)
        if self._on_event:
            self._on_event("done", {"result": fallback})
        return fallback

    # ── 旧版兼容 ────────────────────────────────────────────────────

    async def _run_legacy(
        self, task, max_rounds, verbose, on_round_complete, on_event
    ) -> str:
        """旧版轮次循环（Planner 不可用时的降级方案）"""
        self._on_event = on_event
        full_history = f"原始任务：{task}\n\n"
        all_summaries: list[SummaryResult] = []

        for round_num in range(1, max_rounds + 1):
            if self._on_event:
                self._on_event("round_start", {"round": round_num, "mode": "legacy"})
            if verbose:
                self._print_separator()
                self._print(f"第 {round_num} 轮开始（旧版模式）", bold=True)

            try:
                decomposition = await self.decompose_task(task, history=full_history)
            except Exception as e:
                if self._on_event:
                    self._on_event("error", {"message": f"任务分解失败: {e}"})
                break

            phases_set = sorted(set(st.phase for st in decomposition.subtasks))
            subtasks_info = [
                {"worker": st.worker, "description": st.description, "phase": st.phase}
                for st in decomposition.subtasks
            ]
            if self._on_event:
                self._on_event("phase", {"phase": "decompose", "status": "done",
                    "analysis": decomposition.analysis, "subtasks": subtasks_info, "phases": phases_set})

            if not decomposition.subtasks:
                break

            results, agent_steps_map = await self.execute_phased(decomposition.subtasks)
            round_result = RoundResult(
                round_num=round_num,
                subtask_results=results,
                subtask_details=[(st.worker, st.description, st.phase) for st in decomposition.subtasks],
                phase_count=len(phases_set),
                agent_steps=agent_steps_map,
            )
            summary = await self.summarize_round(task, round_result)

            round_text = f"--- 第 {round_num} 轮 ---\n分析：{decomposition.analysis}\n"
            for st in decomposition.subtasks:
                round_text += f"  - Phase {st.phase}[{st.worker}] {st.description}\n"
            round_text += f"总结：{summary.summary}\n\n"
            full_history += round_text
            all_summaries.append(summary)

            if on_round_complete:
                on_round_complete(round_num, decomposition, round_result, summary)
            if self._on_event:
                self._on_event("round_done", {"round": round_num, "is_final": summary.is_final})

            if summary.is_final:
                result = summary.final_answer or summary.summary
                if self._on_event:
                    self._on_event("done", {"result": result})
                return result

        if all_summaries:
            last = all_summaries[-1]
            if last.final_answer:
                if self._on_event:
                    self._on_event("done", {"result": last.final_answer})
                return last.final_answer
        return last.summary if all_summaries else "未能完成任何轮次处理。"

    # ── 辅助方法 ────────────────────────────────────────────────────

    def _build_fallback_answer(
        self,
        workflow: Workflow,
        step_results: dict[str, str],
    ) -> str:
        """当评估未通过但已超过迭代次数时，综合各步骤输出生成最终答案"""
        parts = [f"# 综合报告\n\n> 工作流分析：{workflow.analysis}\n"]
        for phase_num in sorted(workflow.phases.keys()):
            for step in workflow.phases[phase_num]:
                content = step_results.get(step.step_id, "[未执行]")
                parts.append(f"\n## {step.worker} — {step.description}\n\n{content}\n")
        return "\n".join(parts)

    # ── 输出工具 ───────────────────────────────────────────────────────

    @staticmethod
    def _print(text: str, color: str = "", bold: bool = False):
        codes = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", "blue": "\033[94m", "cyan": "\033[96m"}
        reset = "\033[0m"
        prefix = codes.get(color, "")
        boldc = "\033[1m" if bold else ""
        print(f"{boldc}{prefix}{text}{reset}")

    @staticmethod
    def _print_separator():
        print("\n" + "─" * 60)
