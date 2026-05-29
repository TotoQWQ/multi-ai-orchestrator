"""
multi_ai_system / planner.py
工作流规划器 —— 第一个 AI 分析任务并生成完整执行计划，
后续 Worker 照计划执行，执行后评估质量，不合格则重新规划。

架构：
  Planner（规划 AI）
    → 生成 Workflow（所有阶段、步骤、验收标准）
    → Workers 按 Workflow 执行
    → Evaluator（同一规划 AI）评估质量
    → 不合格 → Planner 修正 Workflow 重新执行
    → 合格 → 输出最终答案
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator import WorkerConfig


# ---------------------------------------------------------------------------
# JSON 解析工具（自包含，避免与 orchestrator 循环导入）
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
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    """工作流中的一个步骤"""
    step_id: str               # 唯一标识，如 "step_0_architect"
    worker: str                # 分配给哪个 Worker
    phase: int                 # 执行阶段（同 phase 并行）
    description: str           # 任务描述
    expected_output: str = ""  # 预期产出（给 Worker 明确的目标）
    depends_on: list[str] | None = None  # 依赖的 step_id 列表


@dataclass
class Workflow:
    """一个完整的工作流计划"""
    task: str                  # 原始任务
    analysis: str              # 规划分析（为什么这样安排）
    phases: dict[int, list[WorkflowStep]]  # phase → steps
    total_steps: int
    quality_threshold: float = 7.0  # 质量分数阈值（0-10，低于此值需重新规划）


@dataclass
class WorkflowEvaluation:
    """工作流执行后的质量评估"""
    score: float               # 0-10 质量评分
    is_satisfactory: bool      # 是否通过
    issues: list[str]          # 发现的问题
    suggestions: str           # 改进建议（给 Planner 用于修正工作流）
    final_answer: str = ""     # 通过时的最终答案


@dataclass
class PlanRoundResult:
    """规划-执行-评估 一个迭代的结果（替代原来的 RoundResult）"""
    iteration: int
    workflow: Workflow
    step_results: dict[str, str]     # step_id → 输出
    agent_steps: dict[str, list[dict]]  # worker_name → agent 步骤
    evaluation: WorkflowEvaluation


# ---------------------------------------------------------------------------
# 规划器
# ---------------------------------------------------------------------------

class WorkflowPlanner:
    """
    工作流规划 AI。
    负责两件事：
      1. plan()  — 分析任务，生成完整工作流
      2. evaluate() — 评估执行结果，决定是否通过
    """

    def __init__(self, client, workers: list["WorkerConfig"]):
        self.client = client
        self.workers = workers
        self.workers_map = {w.name: w for w in workers}
        self._on_event = None

    def set_event_callback(self, cb):
        self._on_event = cb

    def _emit(self, event_type: str, data: dict):
        if self._on_event:
            try:
                self._on_event(event_type, data)
            except Exception:
                pass

    # ── 规划提示词 ──────────────────────────────────────────────────

    def _build_planner_system_prompt(self) -> str:
        worker_lines = "\n".join(
            f"  - {w.name}：{w.description}" + (" [Agent模式，可用工具: " + ", ".join(w.tools) + "]" if w.agent_mode else "")
            for w in self.workers
        )
        return f"""你是一个工作流规划专家（Planner），使用 DeepSeek 模型。

## 职责
分析用户任务，设计一个多 AI 协作的完整工作流，确保覆盖所有环节、产出高质量结果。

## 可用的专职 AI
{worker_lines}

## 工作流设计原则
1. **分阶段（phase）**：同 phase 步骤并行执行，不同 phase 串行
2. **明确产出**：每步写明 expected_output，让 Worker 清楚知道要产出什么
3. **合理依赖**：后续步骤通过 depends_on 引用前序步骤
4. **覆盖全面**：确保工作流覆盖任务的所有关键方面
5. **质量可控**：每步设置 quality_check 验收标准

## 典型工作流模式
- 分析设计类：架构师(phase0) → [审查员, 安全, 测试](phase1, 并行) → 文档(phase2)
- 代码开发类：架构师(phase0) → [开发, 测试](phase1) → 审查(phase2) → 文档(phase3)
- 内容创作类：[创意作家, 文学评论家](phase0) → 综合编辑(phase1)

## 输出格式（严格 JSON）
{{
  "analysis": "工作流设计分析：为什么这样安排各阶段和 Worker",
  "quality_threshold": 7.0,
  "steps": [
    {{
      "step_id": "step_0_architect",
      "worker": "架构设计师",
      "phase": 0,
      "description": "设计微服务电商平台的系统架构",
      "expected_output": "完整的架构文档：技术选型、模块划分、接口定义、数据流图",
      "depends_on": null
    }},
    {{
      "step_id": "step_1_review",
      "worker": "代码审查员",
      "phase": 1,
      "description": "审查架构方案的合理性和潜在风险",
      "expected_output": "审查报告：发现的问题列表和优化建议",
      "depends_on": ["step_0_architect"]
    }}
  ]
}}

depends_on 为 null 或不填表示无依赖（看到所有前序输出）。
quality_threshold 为质量阈值（0-10），低于此分数需重新规划。"""

    # ── 规划 ────────────────────────────────────────────────────────

    async def plan(
        self,
        task: str,
        previous_results: str = "",
        previous_issues: str = "",
    ) -> Workflow:
        """生成工作流计划。如有前次执行结果和问题，则生成修正版工作流。"""
        self._emit("plan_start", {"iteration": 0})

        if previous_results:
            user_msg = (
                f"== Re-Plan ==\n"
                f"原始任务：{task}\n\n"
                f"上次工作流执行结果：\n{previous_results[:3000]}\n\n"
                f"发现的问题：\n{previous_issues}\n\n"
                f"请根据问题和执行结果，设计一个修正后的工作流。"
                f"调整缺少的步骤、替换不合适的 Worker、优化阶段划分。"
                f"输出 JSON 格式。"
            )
        else:
            user_msg = (
                f"== Plan ==\n"
                f"请为以下任务设计一个完整的多 AI 协作工作流：\n\n{task}\n\n"
                f"输出 JSON 格式的工作流计划。"
            )

        messages = [
            {"role": "system", "content": self._build_planner_system_prompt()},
            {"role": "user", "content": user_msg},
        ]

        response = await self._call_llm(messages)
        data = _extract_json(response)

        # 解析步骤
        steps: list[WorkflowStep] = []
        for st in data.get("steps", []):
            deps = st.get("depends_on")
            if isinstance(deps, list):
                deps = [d.strip() for d in deps if isinstance(d, str)]
            else:
                deps = None

            # 验证 Worker 存在
            worker = st.get("worker", "")
            if worker not in self.workers_map:
                worker = self._find_closest_worker(worker)

            steps.append(WorkflowStep(
                step_id=st.get("step_id", f"step_{len(steps)}"),
                worker=worker,
                phase=int(st.get("phase", 0)),
                description=st.get("description", ""),
                expected_output=st.get("expected_output", ""),
                depends_on=deps,
            ))

        if not steps:
            raise ValueError("工作流规划失败：未生成任何步骤。\n" + response[:500])

        # 组织 phases
        phases: dict[int, list[WorkflowStep]] = {}
        for st in steps:
            phases.setdefault(st.phase, []).append(st)

        workflow = Workflow(
            task=task,
            analysis=data.get("analysis", ""),
            phases=phases,
            total_steps=len(steps),
            quality_threshold=float(data.get("quality_threshold", 7.0)),
        )

        self._emit("plan_done", {
            "analysis": workflow.analysis,
            "total_steps": workflow.total_steps,
            "phases": sorted(workflow.phases.keys()),
            "steps": [
                {"step_id": s.step_id, "worker": s.worker, "phase": s.phase, "description": s.description[:100]}
                for s in steps
            ],
        })

        return workflow

    # ── 评估 ────────────────────────────────────────────────────────

    async def evaluate(
        self,
        task: str,
        workflow: Workflow,
        step_results: dict[str, str],
    ) -> WorkflowEvaluation:
        """评估工作流执行结果的质量"""
        self._emit("evaluate_start", {})

        # 构建评估上下文
        results_summary_parts = []
        for phase_num in sorted(workflow.phases.keys()):
            for st in workflow.phases[phase_num]:
                content = step_results.get(st.step_id, "[无结果]")
                preview = content[:600] if len(content) > 600 else content
                results_summary_parts.append(
                    f"### Phase {st.phase} [{st.worker}] {st.step_id}\n"
                    f"预期产出：{st.expected_output}\n"
                    f"实际产出：{preview}\n"
                )
        results_text = "\n".join(results_summary_parts)

        eval_prompt = f"""你是一个质量评估专家。请评估以下工作流的执行结果。

原始任务：{task}

工作流结构：{workflow.analysis}

各步骤执行结果：
{results_text}

## 评估维度
1. **完整性**：所有步骤都执行了吗？产出是否符合 expected_output？
2. **质量**：输出内容的深度、准确性、实用性如何？
3. **一致性**：不同 Worker 的输出是否存在矛盾？
4. **覆盖度**：原始任务的所有方面是否都被覆盖？

## 输出格式（严格 JSON）
{{
  "score": 8.5,
  "issues": ["问题1", "问题2"],
  "suggestions": "改进建议：哪些步骤需要调整、增加或替换 Worker",
  "final_answer": "如果通过（score >= {workflow.quality_threshold}），输出完整的最终答案（Markdown）；否则留空"
}}

评分标准：
- 0-4：严重缺陷，必须重新规划
- 5-6：有明显不足，建议重新规划
- 7-8：基本合格，可以接受
- 9-10：优秀，完美覆盖"""

        messages = [
            {"role": "system", "content": "你是一个严谨的质量评估专家。请客观、公正地评估工作流执行结果。"},
            {"role": "user", "content": eval_prompt},
        ]

        response = await self._call_llm(messages, temperature=0.3)
        data = _extract_json(response)

        score = float(data.get("score", 5))
        threshold = workflow.quality_threshold
        is_satisfactory = score >= threshold

        evaluation = WorkflowEvaluation(
            score=score,
            is_satisfactory=is_satisfactory,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", ""),
            final_answer=data.get("final_answer", "") if is_satisfactory else "",
        )

        self._emit("evaluate_done", {
            "score": evaluation.score,
            "is_satisfactory": evaluation.is_satisfactory,
            "issues": evaluation.issues,
            "suggestions": evaluation.suggestions[:200],
        })

        return evaluation

    # ── 辅助方法 ────────────────────────────────────────────────────

    async def _call_llm(self, messages: list[dict], temperature: float = 0.7) -> str:
        """调用 LLM（带 SSE 流式推送）"""
        if self._on_event:
            def on_token(rea, txt, full):
                if rea:
                    self._on_event("reasoning", {"text": rea})
                if txt:
                    self._on_event("token", {"text": txt})
            return await self.client.chat(messages, temperature, on_token=on_token)
        return await self.client.chat(messages, temperature)

    def _find_closest_worker(self, name: str) -> str:
        """模糊匹配 Worker 名称，找不到则返回第一个"""
        name_lower = name.strip().lower()
        # 精确匹配
        if name in self.workers_map:
            return name
        # 子串匹配
        for w in self.workers:
            if name_lower in w.name.lower() or w.name.lower() in name_lower:
                return w.name
        # 兜底
        return self.workers[0].name if self.workers else name
