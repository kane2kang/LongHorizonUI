"""
LongHorizonUI - 统一入口脚本
支持两种运行模式：
  1. offline  — 基于本地 UI 截图序列的模拟执行（无需连接手机）
  2. live     — 通过 USB 连接真实 Android 设备执行

支持两种指令级别：
  - high  — 仅提供高级任务描述，Agent 自主规划步骤
  - low   — 提供详细的分步操作指令
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from LonghorizonAgent.common import llm_provider
from LonghorizonAgent.agent.auto_execution_agent import AutoExecutionAgent, AutoExecutionConfig
from LonghorizonAgent.system.android_context import AndroidContext, AndroidContextConfig
from LonghorizonAgent.controller.android_controller import AndroidController
from LonghorizonAgent.prompts.auto_execution_prompt import AndroidExecSystemPrompt, AutoExecAgentPrompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("LongHorizonUI")


# ─────────────────────────────────────────
# LLM 初始化
# ─────────────────────────────────────────
def init_llm(provider: str, model: str) -> llm_provider.LLMProvider:
    """根据 provider 名称初始化 LLM 客户端。"""
    if provider == "gemini":
        project = os.getenv("GOOGLE_PROJECT", "")
        location = os.getenv("GOOGLE_LOCATION", "")
        google_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        # 如果是相对路径，转为基于项目根目录的绝对路径
        if google_key_path and not os.path.isabs(google_key_path):
            google_key_path = str(Path(__file__).resolve().parent / google_key_path)
        return llm_provider.LLMProvider(
            llm_provider="gemini", model=model,
            project=project, location=location,
            google_key_json_path=google_key_path if google_key_path else None
        )
    elif provider == "azure_openai":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        return llm_provider.LLMProvider(
            llm_provider="azure_openai", model=model,
            base_url=endpoint, api_key=api_key
        )
    elif provider == "openai":
        endpoint = os.getenv("OPENAI_ENDPOINT", "")
        api_key = os.getenv("OPENAI_API_KEY", "")
        return llm_provider.LLMProvider(
            llm_provider="openai", model=model,
            base_url=endpoint, api_key=api_key
        )
    else:
        raise ValueError(f"不支持的 LLM provider: {provider}，可选: gemini / azure_openai / openai")


# ─────────────────────────────────────────
# 任务加载
# ─────────────────────────────────────────
def load_task(task_dir: Path, instruction_level: str):
    """
    从 task_infos.json 中加载任务信息。

    Args:
        task_dir: 任务目录（包含 task_infos.json 和 screenshot/ 子目录）
        instruction_level: "high" 或 "low"

    Returns:
        (task_name, task_steps)
    """
    task_infos_path = task_dir / "task_infos.json"
    if not task_infos_path.exists():
        raise FileNotFoundError(f"未找到 {task_infos_path}")

    with open(task_infos_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    task_name = task_data.get("task_name", "未命名任务")

    if instruction_level == "low":
        # LOW 模式：加载详细步骤
        steps = task_data.get("task_steps", [])
        task_steps = "\n".join(
            [f"{i + 1}. {step['action']}" for i, step in enumerate(steps)]
        )
        logger.info(f"[LOW 模式] 加载 {len(steps)} 个详细步骤")
    else:
        # HIGH 模式：仅使用任务描述
        task_steps = ""
        logger.info(f"[HIGH 模式] 仅加载任务描述: {task_name}")

    return task_name, task_steps


# ─────────────────────────────────────────
# 模式 1: Offline — 基于本地截图模拟执行
# ─────────────────────────────────────────
def run_offline(args):
    """基于本地截图序列的模拟执行模式。"""
    llm = init_llm(args.provider, args.model)
    data_root = Path(args.data_dir)

    if not data_root.exists():
        logger.error(f"数据目录不存在: {data_root}")
        return

    # 遍历数据目录下的所有任务子目录
    task_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if not task_dirs:
        logger.error(f"在 {data_root} 中未找到任务子目录")
        return

    output_base = Path(args.output_dir)
    logger.info(f"共发现 {len(task_dirs)} 个任务目录")

    for task_dir in task_dirs:
        screenshot_dir = task_dir / "screenshot"
        if not screenshot_dir.exists() or not any(screenshot_dir.glob("*.png")):
            logger.warning(f"跳过 {task_dir.name}：截图目录为空或不存在")
            continue

        task_infos_path = task_dir / "task_infos.json"
        if not task_infos_path.exists():
            logger.warning(f"跳过 {task_dir.name}：未找到 task_infos.json")
            continue

        try:
            task_name, task_steps = load_task(task_dir, args.instruction_level)
        except Exception as e:
            logger.error(f"加载任务 {task_dir.name} 失败: {e}")
            continue

        task_output_dir = output_base / task_dir.name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'=' * 50}")
        logger.info(f"任务: {task_name}")
        logger.info(f"截图来源: {screenshot_dir}")
        logger.info(f"输出目录: {task_output_dir}")
        logger.info(f"{'=' * 50}")

        # 配置系统上下文（本地截图模式）
        context_config = AndroidContextConfig(
            local_screenshot_dir=str(screenshot_dir),
            use_perception=True,
            use_ocr=True,
            use_ocr_rec=False,
            perception_description_type="normal",
            highlight_grid_num=24,
            screenshot_save_dir=str(task_output_dir),
            detect_split_x=2,
            detect_split_y=2,
        )
        system_context = AndroidContext(config=context_config)
        controller = AndroidController(highlight_action=True)

        # 配置 Agent
        agent_config = AutoExecutionConfig(
            max_steps=args.max_steps,
            llm_temperature=args.temperature,
            max_input_len=60,
            keep_last_n_states=2,
        )
        agent = AutoExecutionAgent(
            agent_config=agent_config,
            controller=controller,
            system_context=system_context,
            llm=llm,
            system_prompt_class=AndroidExecSystemPrompt,
            agent_prompt_class=AutoExecAgentPrompt,
            task_dir=task_dir,
        )

        # 执行
        try:
            agent_output = agent.run(
                task=task_name,
                task_steps=task_steps,
                task_infos=f"原始任务目录: {task_dir.name}"
            )
            logger.info(f"✅ 任务完成，输出保存在: {agent_output}")
        except Exception as e:
            import traceback
            logger.error(f"❌ 任务执行失败: {e}")
            traceback.print_exc()
            continue


# ─────────────────────────────────────────
# 模式 2: Live — USB 连接真实设备执行
# ─────────────────────────────────────────
def run_live(args):
    """通过 USB 连接真实 Android 设备的执行模式。"""
    import adbutils

    llm = init_llm(args.provider, args.model)

    # 自动检测设备
    devices = adbutils.adb.device_list()
    if not devices:
        logger.error("未检测到 Android 设备，请确认：\n"
                      "  1. 已通过 USB 连接设备\n"
                      "  2. 已开启 USB 调试\n"
                      "  3. 运行 'adb devices' 能看到设备")
        return

    device_serial = args.device or devices[0].info["serialno"]
    logger.info(f"使用设备: {device_serial}")

    output_base = Path(args.output_dir)

    # 如果提供了任务文件，从文件加载
    if args.task_file:
        task_file = Path(args.task_file)
        if not task_file.exists():
            logger.error(f"任务文件不存在: {task_file}")
            return
        with open(task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        task_name = task_data.get("task_name", "未命名任务")
        if args.instruction_level == "low":
            steps = task_data.get("task_steps", [])
            task_steps = "\n".join(
                [f"{i + 1}. {step['action']}" for i, step in enumerate(steps)]
            )
        else:
            task_steps = ""
    else:
        # 交互式输入
        task_name = args.task or input("请输入任务描述: ").strip()
        task_steps = ""
        if args.instruction_level == "low":
            print("请输入详细步骤（每行一步，输入空行结束）：")
            step_lines = []
            while True:
                line = input().strip()
                if not line:
                    break
                step_lines.append(line)
            task_steps = "\n".join(step_lines)

    if not task_name:
        logger.error("任务描述不能为空")
        return

    logger.info(f"\n{'=' * 50}")
    logger.info(f"任务: {task_name}")
    logger.info(f"指令级别: {args.instruction_level}")
    logger.info(f"设备: {device_serial}")
    logger.info(f"{'=' * 50}")

    # 配置系统上下文（真实设备模式）
    context_config = AndroidContextConfig(
        device_id=device_serial,
        use_perception=True,
        use_ocr=True,
        use_ocr_rec=False,
        perception_description_type="normal",
        highlight_grid_num=24,
        screenshot_save_dir=str(output_base / f"live_{device_serial}"),
        detect_split_x=2,
        detect_split_y=2,
    )
    system_context = AndroidContext(config=context_config)
    controller = AndroidController(highlight_action=True)

    # 配置 Agent
    agent_config = AutoExecutionConfig(
        max_steps=args.max_steps,
        llm_temperature=args.temperature,
        max_input_len=60,
        keep_last_n_states=2,
    )
    agent = AutoExecutionAgent(
        agent_config=agent_config,
        controller=controller,
        system_context=system_context,
        llm=llm,
        system_prompt_class=AndroidExecSystemPrompt,
        agent_prompt_class=AutoExecAgentPrompt,
    )

    # 执行
    try:
        agent_output = agent.run(
            task=task_name,
            task_steps=task_steps,
            task_infos=f"Live mode on device {device_serial}"
        )
        logger.info(f"✅ 任务完成，输出保存在: {agent_output}")
    except Exception as e:
        import traceback
        logger.error(f"❌ 任务执行失败: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────
# CLI 解析
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LongHorizonUI - 长链路 GUI 自动化 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Offline 模式（截图模拟）- Low 指令
  python run.py offline --data_dir data/general/qq --provider gemini --model gemini-2.5-pro

  # Offline 模式 - High 指令
  python run.py offline --data_dir data/game/hero --instruction_level high --provider gemini --model gemini-2.5-pro

  # Live 模式（USB 连接设备）- 交互式输入
  python run.py live --provider gemini --model gemini-2.5-pro

  # Live 模式 - 从任务文件加载
  python run.py live --task_file data/general/qq/task_001/task_infos.json --provider gemini --model gemini-2.5-pro
        """
    )

    # 公共参数
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["gemini", "azure_openai", "openai"],
                        help="LLM 服务提供商 (默认: gemini)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro",
                        help="模型名称 (默认: gemini-2.5-pro)")
    parser.add_argument("--instruction_level", type=str, default="low",
                        choices=["high", "low"],
                        help="指令级别: high=仅任务描述, low=详细步骤 (默认: low)")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Agent 最大执行步数 (默认: 100)")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="LLM 采样温度 (默认: 0.4)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录 (默认: ./output)")

    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # ── offline 子命令 ──
    offline_parser = subparsers.add_parser("offline", help="基于本地截图的模拟执行模式")
    offline_parser.add_argument("--data_dir", type=str, required=True,
                                help="数据目录路径（包含多个任务子目录，每个子目录含 screenshot/ 和 task_infos.json）")

    # ── live 子命令 ──
    live_parser = subparsers.add_parser("live", help="USB 连接真实 Android 设备执行模式")
    live_parser.add_argument("--device", type=str, default=None,
                             help="指定设备序列号（默认自动检测第一个设备）")
    live_parser.add_argument("--task", type=str, default=None,
                             help="任务描述（不指定则进入交互式输入）")
    live_parser.add_argument("--task_file", type=str, default=None,
                             help="从 task_infos.json 文件加载任务")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    # 加载环境变量
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            logger.info(f"已加载环境变量: {env_path}")
        except ImportError:
            logger.warning("python-dotenv 未安装，请手动设置环境变量或运行: pip install python-dotenv")

    if args.mode == "offline":
        run_offline(args)
    elif args.mode == "live":
        run_live(args)


if __name__ == "__main__":
    main()
