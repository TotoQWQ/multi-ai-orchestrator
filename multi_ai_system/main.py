#!/usr/bin/env python3
"""
multi_ai_system / main.py
CLI 入口 —— 多AI协作编排系统。

使用方式：
  1. 设置环境变量：set DEEPSEEK_API_KEY=你的key
  2. 运行：python main.py
     （交互模式，可反复输入任务）
  3. 或：python main.py "你的大任务"
     （单次模式，直接执行并退出）

配置文件：workers.json（同目录下）
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# 添加当前目录到 sys.path，确保能导入 orchestrator
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import (
    MultiAIOrchestrator,
    WorkerConfig,
)


# ── 配置加载 ──────────────────────────────────────────────────────────────

def load_workers(workers_path: str) -> list[WorkerConfig]:
    """从 JSON 文件加载 Worker 配置"""
    path = Path(workers_path)
    if not path.exists():
        print(f"⚠️  Worker 配置文件不存在：{workers_path}")
        print("   将使用默认的示例 Worker。")
        return _default_workers()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"workers.json 应该是一个 JSON 数组，但得到 {type(data)}")

    workers = []
    for i, item in enumerate(data):
        name = item.get("name", f"助手{i+1}")
        workers.append(WorkerConfig(
            name=name,
            description=item.get("description", ""),
            system_prompt=item.get("system_prompt", "你是一个有用的AI助手。"),
            temperature=float(item.get("temperature", 0.7)),
        ))

    if not workers:
        print("⚠️  Worker 配置文件为空，使用默认配置。")
        return _default_workers()

    return workers


def _default_workers() -> list[WorkerConfig]:
    """默认的示例 Worker 配置"""
    return [
        WorkerConfig(
            name="分析员",
            description="擅长问题分析、需求拆解和信息梳理",
            system_prompt="你是一位严谨的分析专家。请深入分析问题，梳理关键信息，找出核心矛盾和潜在风险。给出清晰的分析框架和结论。",
            temperature=0.6,
        ),
        WorkerConfig(
            name="执行员",
            description="擅长方案设计、代码编写和具体实施",
            system_prompt="你是一位实干型专家。请根据要求给出可落地的方案、代码或操作步骤。注重实用性和可行性，输出的内容要可以直接使用。",
            temperature=0.7,
        ),
        WorkerConfig(
            name="审查员",
            description="擅长质量审查、风险识别和优化建议",
            system_prompt="你是一位质量审查专家。请仔细检查方案或代码，找出潜在问题、安全风险和改进空间。给出具体的优化建议。",
            temperature=0.3,
        ),
    ]


def get_api_key() -> str:
    """获取 DeepSeek API Key"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    # 尝试从 .env 文件读取
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def print_banner():
    banner = """
╔══════════════════════════════════════════════╗
║        🤖 多AI协作编排系统 (Multi-AI)         ║
║        基于 DeepSeek Flash 模型               ║
║   大任务 → 多AI分工 → 并行执行 → 归纳综合     ║
╚══════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    print("可用命令：")
    print("  /quit    退出程序")
    print("  /workers 查看当前专职AI列表")
    print("  /rounds N  设置最大轮数（默认 3）")
    print("  /help    显示此帮助")
    print()


def print_workers(workers: list[WorkerConfig]):
    print(f"\n当前专职AI助手 ({len(workers)} 个)：")
    print(f"  {'名称':<12} {'专长描述'}")
    print(f"  {'────':<12} {'────────'}")
    for w in workers:
        print(f"  {w.name:<12} {w.description}")
    print()


# ── 交互模式 ──────────────────────────────────────────────────────────────

async def interactive_mode(orc: MultiAIOrchestrator):
    """交互式 CLI 循环"""
    print_banner()
    print_workers(orc.workers)
    print_help()

    max_rounds = 3

    while True:
        try:
            user_input = input("\n🔷 请输入任务（或 /help 查看命令）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if not user_input:
            continue

        # ── 内建命令 ──
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("👋 再见！")
                break
            elif cmd == "/help":
                print_help()
                continue
            elif cmd == "/workers":
                print_workers(orc.workers)
                continue
            elif cmd.startswith("/rounds"):
                parts = cmd.split()
                if len(parts) == 2 and parts[1].isdigit():
                    max_rounds = int(parts[1])
                    print(f"✅ 最大轮数设置为 {max_rounds}")
                else:
                    print(f"当前最大轮数：{max_rounds}")
                continue
            else:
                print(f"未知命令：{user_input}，输入 /help 查看可用命令")
                continue

        # ── 执行任务 ──
        print(f"\n{'=' * 60}")
        print(f"📤 开始处理任务（最多 {max_rounds} 轮）")
        print(f"{'=' * 60}\n")

        try:
            result = await orc.run(
                task=user_input,
                max_rounds=max_rounds,
                verbose=True,
            )

            print(f"\n{'=' * 60}")
            print("✅ 最终结果：")
            print(f"{'=' * 60}")
            print(f"\n{result}\n")

        except Exception as e:
            print(f"\n❌ 执行出错：{e}")
            print("  请检查：")
            print("  1. 环境变量 DEEPSEEK_API_KEY 是否正确设置")
            print("  2. 网络是否可访问 api.deepseek.com")
            print("  3. workers.json 格式是否正确")


# ── 单次模式 ──────────────────────────────────────────────────────────────

async def single_mode(orc: MultiAIOrchestrator, task: str):
    """单次执行模式"""
    print_banner()
    print(f"📤 任务：{task}\n")
    result = await orc.run(task=task, max_rounds=3, verbose=True)
    print(f"\n{'=' * 60}")
    print("✅ 最终结果：")
    print(f"{'=' * 60}")
    print(f"\n{result}\n")


# ── 主入口 ────────────────────────────────────────────────────────────────

def main():
    # 1. 获取 API Key
    api_key = get_api_key()
    if not api_key:
        print("❌ 未设置 DEEPSEEK_API_KEY")
        print()
        print("请通过以下任一方式设置：")
        print()
        print("  方式一（推荐）：设置环境变量")
        print("    Windows:  set DEEPSEEK_API_KEY=你的key")
        print("    PowerShell:  $env:DEEPSEEK_API_KEY='你的key'")
        print("    Linux/Mac: export DEEPSEEK_API_KEY=你的key")
        print()
        print("  方式二：在和 main.py 同目录下创建 .env 文件")
        print("    内容：DEEPSEEK_API_KEY=你的key")
        print()
        sys.exit(1)

    # 2. 加载 Worker 配置
    workers_path = str(Path(__file__).parent / "workers.json")
    workers = load_workers(workers_path)

    # 3. 初始化编排器
    orc = MultiAIOrchestrator(
        workers=workers,
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )

    # 4. 执行模式
    if len(sys.argv) > 1:
        # 命令行参数模式：python main.py "任务描述"
        task = " ".join(sys.argv[1:])
        asyncio.run(single_mode(orc, task))
    else:
        # 交互模式
        asyncio.run(interactive_mode(orc))


if __name__ == "__main__":
    main()
