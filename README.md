<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/framework-FastAPI-009688" alt="FastAPI">
  <img src="https://img.shields.io/badge/model-DeepSeek%20Flash-4F46E5" alt="DeepSeek">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/stars-%E2%98%85%E2%98%85%E2%98%85%E2%98%85%E2%98%86-brightgreen" alt="Stars">
</p>

<h1 align="center"> Multi-AI Orchestrator</h1>
<p align="center"><b>大任务 → 多 AI 分工 → 分阶段协作 → 归纳综合</b></p>
<p align="center">基于 DeepSeek Flash 的多 AI 协作编排系统，配备初音未来看板娘</p>

<br>

<p align="center">
  <img src="https://via.placeholder.com/800x450/1a1d2e/818cf8?text=Multi-AI+Orchestrator+Demo" alt="Demo screenshot">
</p>

---

##  特性

| | 特性 | 说明 |
|---|------|------|
| 🧠 | **智能编排** | Orchestrator 自动分析任务、拆分、分配、归纳 |
| 👥 | **12 个专职 AI** | 架构师、审查员、测试、安全、产品经理等各领域专家 |
| ⚡ | **分阶段协作** | 同阶段并行、不同阶段串行，支持 `depends_on` 依赖链 |
| 🔄 | **多轮迭代** | 自动判断是否完成，最多支持 5 轮迭代 |
| 📡 | **SSE 实时流式** | 实时推送执行进度，支持 reasoning + token 级别渲染 |
| 📊 | **可视化流程图** | 实时展示执行流程，可切换日志/流程图双视图 |
| 🌐 | **网页抓取** | 自动/手动抓取网页内容供 AI 分析 |
| 🗣️ | **初音未来看板娘** | 左下角 Live2D 模型，根据执行进度实时汇报 |
| 🔑 | **用户自定义 Key** | 页面内添加/删除多个 API Key，自动轮换 |
| 📝 | **Markdown 渲染** | 结果富文本展示，一键复制纯文本 |
| 📋 | **历史记录** | 每次执行自动保存，支持查看和追问 |
| 💬 | **继续提问** | 任务完成后可基于历史上下文继续追问 |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/TotoQWQ/multi-ai-orchestrator.git
cd multi-ai-orchestrator/multi_ai_system
pip install -r requirements.txt
```

### 启动

**Windows** — 双击 `start.bat` 或在终端执行：

```bash
start.bat
```

**手动启动：**

```bash
python web_app.py
```

### 使用

打开浏览器访问 `http://localhost:5000`

1. ⚙️ 点击右上角 **Keys** → 添加你的 DeepSeek API Key
2. 📝 在输入框中输入大任务描述
3. ▶️ 点击 **Run**，实时查看执行过程
4. 💬 完成后可继续追问或输入 URL 抓取网页

> 💡 **示例任务：**  
> `设计一个微服务电商平台后端，包含订单、支付、库存、用户模块，编写核心代码和测试用例`

---

## 🏗️ 架构设计

```
┌─ 用户输入大任务 ─────────────────────────────┐
                                               │
  ┌─────────────────────────────────────────┐   │
  │  🧠 Orchestrator（总规划 AI）            │   │
  │  1. 分析任务，按 phase 拆分子任务        │   │
  │  2. 指定依赖关系（depends_on）           │   │
  └──────────────┬──────────────────────────┘   │
                 │                              │
          ┌──────▼──────┐                       │
          │  Phase 0    │  ← 并行执行            │
          │  Worker A   │                       │
          │  Worker B   │                       │
          └──────┬──────┘                       │
                 │ 前序输出自动注入上下文          │
          ┌──────▼──────┐                       │
          │  Phase 1    │  ← depends_on         │
          │  Worker C   │  ← 仅依赖 Worker A    │
          └──────┬──────┘                       │
                 │                              │
          ┌──────▼──────┐                       │
          │  🧠 Orche   │  ← 归纳总结            │
          │  斯特ator    │  ← 判断是否进入下一轮   │
          └─────────────┘                       │
                                               │
└──────────────────────────────────────────────┘
```

### 协作机制

| 阶段 | 执行方式 | 上下文传递 |
|------|---------|-----------|
| **Phase 0** | 🔀 所有任务并行执行 | 无（初始阶段） |
| **Phase 1** | 🔀 仍可并行，但串行于 Phase 0 | 接收到 Phase 0 所有输出 |
| **Phase 2+** | 🔀 同上 | 接收到前序所有输出 |
| **depends_on** | 📍 指定仅依赖特定 Worker | 只传递指定 Worker 的输出 |

---

## 👥 专职 AI 助手

系统内置 12 个不同领域的 AI 专家：

| Worker | 领域 | Temp | 典型任务 |
|--------|------|------|---------|
| 🏗️ **架构设计师** | 系统架构、技术选型 | 0.7 | 设计系统整体架构 |
| 🔍 **代码审查员** | 代码质量、Bug 检测 | 0.3 | 审查代码安全性 |
| 🧪 **测试工程师** | 测试策略、用例设计 | 0.5 | 设计测试方案 |
| 📖 **文档撰写员** | 技术文档、API 文档 | 0.6 | 编写开发文档 |
| 📊 **数据分析师** | 数据分析、可视化 | 0.5 | 数据洞察报告 |
| 🛡️ **安全工程师** | 安全审计、合规 | 0.4 | 安全漏洞扫描 |
| 🎨 **UI/UX 设计师** | 界面设计、交互 | 0.8 | 用户体验方案 |
| 📋 **产品经理** | 需求分析、规划 | 0.6 | 功能需求文档 |
| 📚 **文学评论家** | 文学分析、风格评价 | 0.8 | 文本深度解读 |
| ✍️ **创意作家** | 故事创作、文案 | 0.9 | 创意内容撰写 |
| 🌐 **网页信息抓取员** | 网页抓取、提取 | 0.4 | 网页内容分析 |
| 🎤 **看板娘话术师** | 初音未来话术 | 0.9 | 实时进度汇报 |

> 💡 你可以通过编辑 `workers.json` 自定义 Worker 的提示词和参数

---

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/run` | 提交任务 |
| `GET` | `/api/status/{id}` | 查询任务状态 |
| `GET` | `/api/stream/{id}` | SSE 流式推送 |
| `GET` | `/api/workers` | 获取 Worker 列表 |
| `GET` | `/api/history` | 历史记录列表 |
| `GET` | `/api/history/{id}` | 历史记录详情 |
| `POST` | `/api/ask` | 继续提问 |
| `POST` | `/api/fetch` | 抓取网页 |
| `GET` | `/api/keys` | Key 列表 |
| `POST` | `/api/keys` | 添加 Key |
| `DELETE` | `/api/keys/{index}` | 删除 Key |

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| **Python 3.12+** | 运行时 |
| **FastAPI + Uvicorn** | Web 服务 + ASGI |
| **OpenAI SDK** | DeepSeek API 兼容接口 |
| **Jinja2** | 模板渲染 |
| **SSE** | 实时流式推送 |
| **Live2D** | 初音未来看板娘 |
| **marked.js** | Markdown 渲染 |

---

## ⚙️ 配置

通过 `workers.json` 自定义 Worker：

```json
{
  "name": "你的助手名称",
  "description": "简短描述",
  "system_prompt": "系统提示词，定义角色和行为",
  "temperature": 0.7
}
```

支持的环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名称 |
| `PORT` | `5000` | Web 端口 |

---

## 📂 项目结构

```
multi_ai_orchestrator/
├── README.md
├── .gitignore
└── multi_ai_system/
    ├── main.py              # CLI 入口
    ├── web_app.py           # FastAPI Web 服务（585 行）
    ├── orchestrator.py      # 核心编排引擎（671 行）
    ├── workers.json         # 12 个 Worker 配置
    ├── requirements.txt     # 依赖
    ├── start.bat / start.sh # 一键启动
    └── templates/
        └── index.html       # Web 界面（797 行）
```

> 📊 总计约 **2500 行代码**

---

## 📄 License

MIT
