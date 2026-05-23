"""
multi_ai_system / web_app.py
FastAPI Web 应用 —— 为多AI协作编排系统提供浏览器界面。

启动方式：
  python web_app.py
  或
  uvicorn web_app:app --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# 确保能找到 orchestrator 模块
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import (
    MultiAIOrchestrator,
    WorkerConfig,
)

# ---------------------------------------------------------------------------
# 历史记录
# ---------------------------------------------------------------------------

HISTORY_DIR = Path(__file__).parent / "history"


def save_history(task_id: str, task: str, max_rounds: int, data: dict):
    """将任务执行记录保存到 history/ 目录"""
    HISTORY_DIR.mkdir(exist_ok=True)
    entry = {
        "task_id": task_id,
        "task": task,
        "max_rounds": max_rounds,
        "created_at": datetime.now().isoformat(),
        "status": data.get("status", "unknown"),
        "rounds": data.get("rounds", []),
        "final_result": data.get("final_result"),
        "error": data.get("error"),
    }
    path = HISTORY_DIR / f"{task_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


def load_history_list() -> list[dict]:
    """扫描 history/ 目录返回历史记录列表（按时间倒序）"""
    HISTORY_DIR.mkdir(exist_ok=True)
    items = []
    for fpath in sorted(HISTORY_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if fpath.suffix != ".json":
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                entry = json.load(f)
            items.append({
                "task_id": entry.get("task_id", fpath.stem),
                "task": entry.get("task", "")[:120],
                "status": entry.get("status", "unknown"),
                "created_at": entry.get("created_at", ""),
                "rounds": len(entry.get("rounds", [])),
            })
        except Exception:
            continue
    return items


def load_history_detail(task_id: str) -> dict | None:
    """加载单个历史记录详情"""
    path = HISTORY_DIR / f"{task_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时验证配置，关闭时清理资源"""
    # ── startup ──
    try:
        orc = get_orchestrator()
        print(f"编排器初始化成功 ({len(orc.workers)} 个助手)")
        for w in orc.workers:
            print(f"  - {w.name}")
    except Exception as e:
        print(f"[web] 编排器初始化失败: {e}")
        print("[web] 请设置 DEEPSEEK_API_KEY 后重启服务")
    yield
    # ── shutdown ──
    tasks_store.clear()


app = FastAPI(
    title="多AI协作编排系统",
    description="大任务 → 多AI分工 → 并行执行 → 归纳综合",
    lifespan=lifespan,
)

# 模板目录
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# 全局任务状态存储
# ---------------------------------------------------------------------------
# task_id -> {
#   "status": "running" | "completed" | "failed",
#   "rounds": [...],
#   "final_result": str | None,
#   "error": str | None,
#   "total_subtasks": int  (当前轮子任务数，用于前端进度)
# }
tasks_store: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# 初始化编排器（全局单例）
# ---------------------------------------------------------------------------

_orchestrator: MultiAIOrchestrator | None = None


# ── API Key 池（用户通过页面添加，轮询使用） ──
api_key_pool: list[str] = []
_key_index = 0

def get_next_api_key() -> str:
    global _key_index
    pool = api_key_pool or [os.environ.get("DEEPSEEK_API_KEY", "")]
    if not pool or not pool[0]:
        return ""
    key = pool[_key_index % len(pool)]
    _key_index += 1
    return key


def get_orchestrator() -> MultiAIOrchestrator:
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    # Worker 配置
    workers_path = Path(__file__).parent / "workers.json"
    if workers_path.exists():
        with open(workers_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        workers = [
            WorkerConfig(
                name=item.get("name", f"助手{i+1}"),
                description=item.get("description", ""),
                system_prompt=item.get("system_prompt", ""),
                temperature=float(item.get("temperature", 0.7)),
            )
            for i, item in enumerate(data)
        ]
    else:
        # 默认配置
        workers = [
            WorkerConfig(name="架构设计师", description="系统架构设计、模块划分、技术选型",
                         system_prompt="你是一位资深软件架构师。请从架构角度分析问题，关注系统整体的结构、模块划分、接口设计、技术选型和可扩展性。"),
            WorkerConfig(name="代码审查员", description="代码质量审查、bug检测、性能优化",
                         system_prompt="你是一位严谨的代码审查专家。请仔细审查代码，关注：潜在的 bug、性能瓶颈、安全性问题、代码风格和可读性。"),
            WorkerConfig(name="测试工程师", description="测试策略、边界条件分析、测试用例设计",
                         system_prompt="你是一位经验丰富的测试工程师。请设计全面的测试策略，包括：单元测试、集成测试、边界条件、异常场景。"),
            WorkerConfig(name="文档撰写员", description="技术文档、API文档、用户手册编写",
                         system_prompt="你是一位技术文档专家。请以清晰、准确、易懂的方式撰写文档。关注：文档结构、术语统一、示例完整。"),
        ]

    _orchestrator = MultiAIOrchestrator(
        workers=workers,
        api_key="",  # key 由 _run_orchestrator 从池中动态注入
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )
    return _orchestrator


# ---------------------------------------------------------------------------
# Pydantic 模型
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    task: str
    max_rounds: int = 3


class StatusResponse(BaseModel):
    status: str
    rounds: list[dict] = []
    final_result: str | None = None
    error: str | None = None
    total_subtasks: int = 0


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页"""
    error_msg = ""
    workers_data = []
    try:
        orc = get_orchestrator()
        workers_data = [
            {"name": w.name, "description": w.description}
            for w in orc.workers
        ]
    except RuntimeError as e:
        error_msg = str(e)
    except Exception as e:
        error_msg = f"初始化出错: {e}"

    return templates.TemplateResponse(
        request,
        "index.html",
        {"workers": workers_data, "error": error_msg},
    )


@app.post("/api/run")
async def run_task(req: RunRequest):
    """提交任务，返回 task_id"""
    if not req.task.strip():
        return JSONResponse({"error": "任务内容不能为空"}, status_code=400)

    task_id = str(uuid.uuid4())
    tasks_store[task_id] = {
        "status": "running",
        "rounds": [],
        "final_result": None,
        "error": None,
        "total_subtasks": 0,
        "queue": asyncio.Queue(),
    }

    # 在后台启动编排任务
    asyncio.create_task(_run_orchestrator(task_id, req.task, req.max_rounds))

    return {"task_id": task_id}


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态（前端轮询用）"""
    data = tasks_store.get(task_id)
    if data is None:
        return StatusResponse(status="not_found")

    return StatusResponse(
        status=data["status"],
        rounds=data["rounds"],
        final_result=data["final_result"],
        error=data["error"],
        total_subtasks=data.get("total_subtasks", 0),
    )


@app.get("/api/stream/{task_id}")
async def stream_task(task_id: str):
    """SSE 流式推送任务执行进度"""
    data = tasks_store.get(task_id)
    if data is None:
        return JSONResponse({"error": "task not found"}, status_code=404)

    queue: asyncio.Queue = data["queue"]

    async def event_generator():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=600)
                if event["type"] == "done":
                    yield f"event: done\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                    break
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        except asyncio.TimeoutError:
            yield f"event: timeout\ndata: {{}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/workers")
async def get_workers_info():
    """获取专职AI助手列表"""
    try:
        orc = get_orchestrator()
        return [
            {"name": w.name, "description": w.description}
            for w in orc.workers
        ]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history")
async def get_history():
    """获取历史任务列表"""
    return load_history_list()


@app.get("/api/history/{task_id}")
async def get_history_detail(task_id: str):
    """获取单个历史记录详情"""
    detail = load_history_detail(task_id)
    if detail is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return detail


class FetchRequest(BaseModel):
    url: str
    question: str = "总结这个网页的核心内容"


@app.post("/api/fetch")
async def fetch_webpage(req: FetchRequest):
    """抓取网页并用 AI 分析"""
    if not req.url.strip():
        return JSONResponse({"error": "URL 不能为空"}, status_code=400)
    try:
        from orchestrator import MultiAIOrchestrator
        # 临时创建客户端
        orc = get_orchestrator()
        # 抓取
        html = await orc._fetch_url(req.url)
        if html.startswith("[抓取失败"):
            return {"error": html}
        # 用 worker 分析
        talker = orc.workers_map.get("网页信息抓取员")
        if talker:
            msgs = [
                {"role": "system", "content": talker.system_prompt},
                {"role": "user", "content": f"URL: {req.url}\n\n抓取内容：\n{html}\n\n用户需求：{req.question}"},
            ]
            answer = await orc.client.chat(msgs, talker.temperature, max_tokens=2000)
            return {"content": html[:2000], "analysis": answer.strip()}
        # fallback
        return {"content": html[:2000], "analysis": "分析失败"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


class AskRequest(BaseModel):
    task_id: str
    question: str


@app.post("/api/ask")
async def ask_question(req: AskRequest):
    """基于任务历史继续提问"""
    if not req.question.strip():
        return JSONResponse({"error": "问题不能为空"}, status_code=400)

    history = load_history_detail(req.task_id)
    if not history:
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    try:
        orc = get_orchestrator()
        ctx = f"原始任务：{history.get('task', '')}\n\n"
        for r in history.get("rounds", []):
            ctx += f"--- 第{r.get('round_num')}轮 ---\n"
            ctx += f"分析：{r.get('analysis', '')}\n"
            ctx += f"总结：{r.get('summary', '')}\n\n"
        final = history.get("final_result") or history.get("error") or ""
        ctx += f"最终结果：\n{final[:2000]}\n"

        messages = [
            {"role": "system", "content": "你是一个AI助手。基于以下任务执行上下文，回答用户的追问。回答要简短精准，直接解决问题。"},
            {"role": "user", "content": f"{ctx}\n\n用户追问：{req.question}\n\n请回答："},
        ]
        answer = await orc.client.chat(messages, temperature=0.7, max_tokens=2000)
        return {"answer": answer.strip()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API Key 管理 ──────────────────────────────────────────────────────

class AddKeyRequest(BaseModel):
    key: str


@app.get("/api/keys")
async def list_keys():
    """获取已添加的 Key 列表（脱敏）"""
    masked = []
    for i, k in enumerate(api_key_pool):
        masked_val = k[:7] + "..." + k[-4:] if len(k) > 12 else k[:6] + "..."
        masked.append({"index": i, "masked": masked_val, "len": len(k)})
    return {"keys": masked, "count": len(api_key_pool)}


@app.post("/api/keys")
async def add_key(req: AddKeyRequest):
    """添加 API Key"""
    key = req.key.strip()
    if not key:
        return JSONResponse({"error": "Key 不能为空"}, status_code=400)
    if key in api_key_pool:
        return JSONResponse({"error": "Key 已存在"}, status_code=400)
    api_key_pool.append(key)
    return {"index": len(api_key_pool) - 1, "count": len(api_key_pool)}


@app.delete("/api/keys/{index}")
async def remove_key(index: int):
    """删除指定位置的 Key"""
    if index < 0 or index >= len(api_key_pool):
        return JSONResponse({"error": "索引无效"}, status_code=404)
    removed = api_key_pool.pop(index)
    return {"removed": removed[:7] + "...", "count": len(api_key_pool)}


# ---------------------------------------------------------------------------
# 后台编排任务
# ---------------------------------------------------------------------------

async def _run_orchestrator(task_id: str, task: str, max_rounds: int):
    """在后台执行编排器的完整流程"""
    try:
        orc = get_orchestrator()
        # 从 Key 池取一个 key，注入编排器
        key = get_next_api_key()
        if not key:
            raise RuntimeError("请先在 Settings 中添加 API Key")
        orc.set_api_key(key)
        queue: asyncio.Queue = tasks_store[task_id]["queue"]

        # ── 看板娘话术生成（不阻塞主流程） ──
        _talk_lock = False
        async def _push_talk(stage: str, progress: dict = None):
            """progress: 含 subtasks, workers, round, phase, findings 等实时数据"""
            nonlocal _talk_lock
            if _talk_lock:
                return
            _talk_lock = True
            try:
                talker = orc.workers_map.get("看板娘话术师")
                if not talker:
                    return
                prog_text = ""
                if progress:
                    parts = []
                    for k, v in progress.items():
                        if v is not None:
                            parts.append(f"{k}={v}")
                    if parts:
                        prog_text = ", ".join(parts)
                msgs = [
                    {"role": "system", "content": talker.system_prompt + f"\n## 当前阶段\n{stage}"},
                    {"role": "user", "content": f"[进度] {prog_text}\n请生成一句话。" if prog_text else "请生成一句话。"},
                ]
                text = await orc.client.chat(msgs, talker.temperature, max_tokens=80)
                if text.strip():
                    queue.put_nowait({"type": "talk", "data": {"text": text.strip()[:80]}})
            except Exception:
                pass
            finally:
                _talk_lock = False

        # SSE 事件推送回调 —— 同时触发话术生成
        def on_event(event_type: str, data: dict):
            queue.put_nowait({"type": event_type, "data": data})
            # 关键阶段触发话术（携带实时进度数据）
            if event_type == "phase":
                ph = data.get("phase", "")
                st = data.get("status", "")
                if st == "start":
                    asyncio.create_task(_push_talk(ph, {"phase": ph}))
                elif st == "done" and ph == "decompose":
                    subs = data.get("subtasks", [])
                    asyncio.create_task(_push_talk("decompose_done", {"subtasks": len(subs)}))
                elif st == "done" and ph == "summarize":
                    finds = data.get("findings", [])
                    asyncio.create_task(_push_talk("summarize_done", {"findings": len(finds), "is_final": data.get("is_final")}))
            elif event_type == "execute_phase":
                asyncio.create_task(_push_talk("execute", {
                    "phase": data.get("phase_num"), "total_phases": data.get("total"),
                    "workers": len(data.get("workers", [])),
                }))
            elif event_type == "round_start":
                asyncio.create_task(_push_talk("new_round", {"round": data.get("round")}))
            elif event_type == "done":
                asyncio.create_task(_push_talk("complete", {"result_len": len(data.get("result", "") or "")}))
            elif event_type == "error":
                asyncio.create_task(_push_talk("error", {"msg": (data.get("message", "") or "")[:20]}))

        # 轮次完成回调 —— 将详细数据存入全局状态供轮询用
        def on_round(round_num, decomposition, round_result, summary):
            subtasks_info = [
                {"worker": st.worker, "description": st.description, "phase": st.phase, "depends_on": st.depends_on}
                for st in decomposition.subtasks
            ]
            worker_results = {}
            for name, content in round_result.subtask_results.items():
                worker_results[name] = content[:5000] if len(content) > 5000 else content

            round_data = {
                "round_num": round_num,
                "analysis": decomposition.analysis,
                "subtasks": subtasks_info,
                "worker_results": worker_results,
                "summary": summary.summary,
                "key_findings": summary.key_findings,
                "inconsistencies": summary.inconsistencies,
                "is_final": summary.is_final,
                "final_answer": summary.final_answer,
            }
            tasks_store[task_id]["rounds"].append(round_data)
            tasks_store[task_id]["total_subtasks"] = len(subtasks_info)

        # 执行编排（传 on_event 触发流式推送）
        result = await orc.run(
            task=task,
            max_rounds=max_rounds,
            verbose=False,
            on_round_complete=on_round,
            on_event=on_event,
        )

        tasks_store[task_id]["status"] = "completed"
        tasks_store[task_id]["final_result"] = result
        save_history(task_id, task, max_rounds, tasks_store[task_id])

    except Exception as e:
        tasks_store[task_id]["status"] = "failed"
        tasks_store[task_id]["error"] = str(e)
        save_history(task_id, task, max_rounds, tasks_store[task_id])
        # 推送错误事件到 SSE
        try:
            queue: asyncio.Queue = tasks_store[task_id]["queue"]
            queue.put_nowait({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass
        import traceback
        tasks_store[task_id]["error_detail"] = traceback.format_exc()


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"🚀 多AI协作编排系统 Web 服务启动中...")
    print(f"   地址: http://localhost:{port}")
    print(f"   按 Ctrl+C 停止服务")

    uvicorn.run(
        "web_app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
