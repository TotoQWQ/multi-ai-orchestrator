# Multi-AI Orchestrator

大任务 → 多 AI 分工 → 分阶段协作 → 归纳综合

基于 DeepSeek Flash 的多 AI 协作编排系统。Orchestrator（总规划 AI）将用户的大任务拆分为子任务，按阶段（Phase）分配给不同的专职 AI 并行或串行执行，最后归纳总结。支持多轮迭代。

## 架构

```
用户输入大任务
    │
    ▼
┌─────────────────────────────────────┐
│  Orchestrator（总规划 AI）           │
│  1. 分析任务，按 phase 拆分子任务    │
│  2. 指定依赖关系（depends_on）       │
└──────────┬──────────────────────────┘
           │
    ┌──────▼──────┐
    │  Phase 0     │  ← 并行执行
    │  Worker A    │
    │  Worker B    │
    └──────┬──────┘
           │ 前序输出自动注入上下文
    ┌──────▼──────┐
    │  Phase 1     │  ← 可指定 depends_on
    │  Worker C    │  ← 只看到特定 worker 的输出
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Orchestrator│  ← 归纳总结，判断是否进入下一轮
    │  归纳总结    │
    └─────────────┘
```

## 快速开始

### 1. 安装

```bash
cd multi_ai_system
pip install -r requirements.txt
```

### 2. 启动

**Windows：**
```bash
start.bat
```

**手动：**
```bash
python web_app.py
```

### 3. 使用

打开浏览器访问 `http://localhost:5000`。

1. 点击右上角 **⚙ Keys**，添加你的 DeepSeek API Key
2. 在输入框中输入大任务，点击 **Run**
3. 实时查看执行日志和流程图
4. 完成后可继续追问或抓取网页

## 功能特性

### 专职 AI 助手（12 个）

| Worker | 专长 | Temperature |
|--------|------|-------------|
| 架构设计师 | 系统架构、模块划分、技术选型 | 0.7 |
| 代码审查员 | 代码质量、Bug 检测、性能优化 | 0.3 |
| 测试工程师 | 测试策略、边界条件、用例设计 | 0.5 |
| 文档撰写员 | 技术文档、API 文档、用户手册 | 0.6 |
| 数据分析师 | 数据分析、可视化、统计建模 | 0.5 |
| 安全工程师 | 安全审计、漏洞检测、合规 | 0.4 |
| UI/UX 设计师 | 界面设计、用户体验、交互 | 0.8 |
| 产品经理 | 需求分析、功能规划、用户故事 | 0.6 |
| 文学评论家 | 文学分析、文本解读、风格评价 | 0.8 |
| 创意作家 | 故事创作、文案撰写、创意表达 | 0.9 |
| 网页信息抓取员 | 网页抓取、信息提取、在线分析 | 0.4 |
| 看板娘话术师 | 初音未来风格汇报话术 | 0.9 |

### 分阶段协作

Orchestrator 通过 **phase** 组织协作流程：
- 同 phase 的 Worker **并行执行**
- 不同 phase 的 Worker **串行执行**
- 后 phase 自动接收前序输出作为上下文
- 可通过 `depends_on` 指定仅依赖特定 Worker

### 实时流式输出

- SSE 实时推送执行进度
- Token 级别流式渲染（reasoning + content）
- 可切换日志视图 / 流程图视图

### 网页抓取

- Worker 描述中含 URL 时自动抓取
- 完成后可手动输入 URL 抓取并分析

### 初音未来看板娘

- 页面左下角 Live2D 模型
- 根据执行进度自动生成话术
- 撒娇、元气、讨好主人风格

### 用户自定义 API Key

- 页面内添加/删除多个 Key
- 自动轮换使用（round-robin）

### Markdown 渲染

- 最终结果以 Markdown 渲染显示
- 复制功能导出纯文本

### 历史记录

- 每次执行自动保存
- 支持查看和追问

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/run` | 提交任务 |
| GET | `/api/status/{id}` | 查询任务状态 |
| GET | `/api/stream/{id}` | SSE 流式推送 |
| GET | `/api/workers` | 获取 Worker 列表 |
| GET | `/api/history` | 历史记录列表 |
| GET | `/api/history/{id}` | 历史记录详情 |
| POST | `/api/ask` | 继续提问 |
| POST | `/api/fetch` | 抓取网页 |
| GET | `/api/keys` | Key 列表 |
| POST | `/api/keys` | 添加 Key |
| DELETE | `/api/keys/{index}` | 删除 Key |

## 配置

通过 `workers.json` 自定义 Worker：
- `name` — 名称
- `description` — 专长描述
- `system_prompt` — 系统提示词
- `temperature` — 创造性（0.0-1.0）

## 技术栈

- Python 3.12+
- FastAPI + Uvicorn
- OpenAI SDK（DeepSeek 兼容）
- Jinja2 + HTML/CSS/JS
- SSE 实时推送
- Live2D 看板娘
