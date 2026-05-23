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


@dataclass
class SubTask:
    """一个子任务"""
    worker: str               # 分配给哪个助手（name）
    description: str          # 任务描述
    phase: int = 0            # 执行阶段：同 phase 并行，不同 phase 串行
    depends_on: list[str] | None = None  # 依赖的 worker 列表，仅看到这些 worker 的输出


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

    async def _execute_one(self, subtask: SubTask, context: str = "") -> tuple[str, str]:
        """执行单个子任务，可附带前序阶段的上下文。自动抓取描述中的 URL"""
        async with self._semaphore:
            worker = self.workers_map.get(subtask.worker)
            if not worker:
                return (
                    subtask.worker,
                    f"[错误] 找不到名为 '{subtask.worker}' 的助手。可用：{list(self.workers_map.keys())}",
                )

            if self._on_event:
                self._on_event("worker_start", {"worker": subtask.worker, "description": subtask.description[:120], "phase": subtask.phase})

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

            messages = [
                {"role": "system", "content": worker.system_prompt},
                {"role": "user", "content": f"== Execute ==\n{full_prompt}"},
            ]

            result = await self._call_llm(messages, worker.temperature)

            if self._on_event:
                self._on_event("worker_done", {"worker": subtask.worker, "phase": subtask.phase, "summary": result[:200]})
            return subtask.worker, result

    async def execute_phased(self, subtasks: list[SubTask]) -> dict[str, str]:
        """
        分阶段协作执行：
        - 先按 phase 分组
        - 同 phase 并行执行
        - 不同 phase 串行执行，后 phase 看到前序所有输出
        """
        # 按 phase 分组
        phases: dict[int, list[SubTask]] = {}
        for st in subtasks:
            phases.setdefault(st.phase, []).append(st)

        all_results: dict[str, str] = {}
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
                if isinstance(res, Exception):
                    all_results[phase_tasks[i].worker] = f"[执行异常] {res}"
                else:
                    worker_name, content = res
                    all_results[worker_name] = content

        if self._on_event:
            self._on_event("phase", {"phase": "execute", "status": "done"})

        return all_results

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

    # ── 主编排流程 ─────────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        max_rounds: int = 3,
        verbose: bool = True,
        on_round_complete=None,
        on_event=None,
    ) -> str:
        """
        主编排流程：
          1. Orchestrator 分析任务，按 phase 拆分为子任务
          2. 分阶段协作执行（同 phase 并行，不同 phase 串行）
          3. Orchestrator 归纳总结
          4. 判断是否完成，否则回到步骤1（最多 max_rounds 轮）
        """
        self._on_event = on_event
        full_history = f"原始任务：{task}\n\n"
        all_summaries: list[SummaryResult] = []

        for round_num in range(1, max_rounds + 1):
            if self._on_event:
                self._on_event("round_start", {"round": round_num})
            if verbose:
                self._print_separator()
                self._print(f"第 {round_num} 轮开始", bold=True)

            # ── 1. 分解 ──
            if self._on_event:
                self._on_event("phase", {"phase": "decompose", "status": "start", "round": round_num})
            if verbose:
                self._print("正在分析并分解任务…")
            try:
                decomposition = await self.decompose_task(task, history=full_history)
            except Exception as e:
                if self._on_event:
                    self._on_event("error", {"message": f"任务分解失败: {e}"})
                self._print(f"任务分解失败：{e}", color="red")
                break

            # 按 phase 分组统计
            phases_set = sorted(set(st.phase for st in decomposition.subtasks))
            subtasks_info = [{"worker": st.worker, "description": st.description, "phase": st.phase} for st in decomposition.subtasks]

            if self._on_event:
                self._on_event("phase", {"phase": "decompose", "status": "done", "analysis": decomposition.analysis, "subtasks": subtasks_info, "phases": phases_set})

            if verbose:
                self._print(f"分析：{decomposition.analysis}")
                self._print(f"协作方案：{len(phases_set)} 个阶段，{len(decomposition.subtasks)} 个子任务")
                for phase in phases_set:
                    phase_tasks = [st for st in decomposition.subtasks if st.phase == phase]
                    self._print(f"  Phase {phase}（{'并行' if len(phase_tasks) > 1 else '单任务'}）：")
                    for st in phase_tasks:
                        self._print(f"    ├─ [{st.worker}] {st.description[:100]}…")

            if not decomposition.subtasks:
                if verbose:
                    self._print("没有分解出子任务，结束循环", color="yellow")
                break

            # ── 2. 分阶段协作执行 ──
            if verbose:
                self._print(f"\n开始分阶段协作执行…")
            results = await self.execute_phased(decomposition.subtasks)

            if verbose:
                self._print(f"所有阶段执行完成：")
                for name, content in results.items():
                    preview = content[:120].replace("\n", " ")
                    self._print(f"  [{name}] {preview}…")

            # ── 3. 总结 ──
            if self._on_event:
                self._on_event("phase", {"phase": "summarize", "status": "start"})
            if verbose:
                self._print(f"\n正在归纳总结…")

            round_result = RoundResult(
                round_num=round_num,
                subtask_results=results,
                subtask_details=[(st.worker, st.description, st.phase) for st in decomposition.subtasks],
                phase_count=len(phases_set),
            )
            summary = await self.summarize_round(task, round_result)

            if self._on_event:
                self._on_event("phase", {"phase": "summarize", "status": "done", "summary": summary.summary, "findings": summary.key_findings, "is_final": summary.is_final})

            if verbose:
                self._print(f"\n第 {round_num} 轮总结：")
                self._print(f"  {summary.summary[:200]}…")

            # ── 4. 记录 + 回调 ──
            round_text = f"--- 第 {round_num} 轮 ---\n分析：{decomposition.analysis}\n"
            for st in decomposition.subtasks:
                round_text += f"  - Phase {st.phase}[{st.worker}] {st.description}\n"
            round_text += f"总结：{summary.summary}\n"
            round_text += f"关键发现：{'；'.join(summary.key_findings)}\n"
            round_text += f"是否完成：{'是' if summary.is_final else '否'}\n\n"
            full_history += round_text
            all_summaries.append(summary)

            if on_round_complete:
                on_round_complete(round_num, decomposition, round_result, summary)
            if self._on_event:
                self._on_event("round_done", {"round": round_num, "is_final": summary.is_final, "summary": summary.summary[:200]})

            # ── 5. 判断 ──
            if summary.is_final:
                if verbose:
                    self._print(f"任务已完成！", bold=True)
                result = summary.final_answer or summary.summary
                if self._on_event:
                    self._on_event("done", {"result": result})
                return result

            if round_num < max_rounds and verbose:
                self._print(f"进入第 {round_num + 1} 轮，焦点：{summary.next_round_focus}")

        # ── 所有轮次结束 ──
        if not all_summaries:
            msg = "未能完成任何轮次处理。"
            if self._on_event:
                self._on_event("done", {"result": msg})
            return msg

        last = all_summaries[-1]
        if last.final_answer:
            if self._on_event:
                self._on_event("done", {"result": last.final_answer})
            return last.final_answer

        if verbose:
            self._print("\n生成最终综合报告…")
        if self._on_event:
            self._on_event("phase", {"phase": "final_synthesis", "status": "start"})

        try:
            final = await self._final_synthesis(task, full_history)
            if self._on_event:
                self._on_event("done", {"result": final})
            return final
        except Exception:
            fallback = last.summary
            if self._on_event:
                self._on_event("done", {"result": fallback})
            return fallback

    # ── 最终综合 ────────────────────────────────────────────────────────

    async def _final_synthesis(self, original_task: str, history: str) -> str:
        messages = [
            {"role": "system", "content": self.orchestrator_system_prompt},
            {
                "role": "user",
                "content": (
                    f"所有轮次的协作已经完成。\n\n"
                    f"原始任务：{original_task}\n\n"
                    f"执行历史：\n{history}\n\n"
                    "请输出最终的综合答案。格式：\n"
                    '{"final_answer": "完整详尽的最终答案（Markdown 格式，结构鲜明）"}'
                ),
            },
        ]
        response = await self._call_llm(messages)
        data = _extract_json(response)
        return data.get("final_answer", response)

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
