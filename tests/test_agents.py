import pdb
import os
import json
import numpy as np
import logging
import sys
from pathlib import Path
import adbutils
from adbutils import AdbError
from pathlib import Path
import json
import time

sys.path.append(".")


def test_record_to_simple_steps():
    from LonghorizonAgent.agent.record_to_simple_steps_agent import RecordToSimpleStepsAgent
    import logging

    from LonghorizonAgent.common import llm_provider, utils
    from LonghorizonAgent.memory.operation_graph import OperationGraph
    import os

    model = "gemini-2.5-pro-preview-05-06"
    # model = "gemini-2.0-flash"
    project = os.getenv("GOOGLE_PROJECT", "")
    location = os.getenv("GOOGLE_LOCATION", "")
    llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location)

    # AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    # MODEL = "gpt-4o"
    # llm = llm_provider.LLMProvider(llm_provider="azure_openai", model=MODEL, base_url=AZURE_OPENAI_ENDPOINT,
    #                                api_key=AZURE_OPENAI_API_KEY)
    # main_dir = "data/mdnf_raw"
    # 多应用场景
    app_scenarios = [
        # "data/general/qq",
        # "data/general/qq_mail",
        # "data/general/qq_music",
        # "data/general/tencent_meeting",
        # "data/general/weixin_new",
        # "data/general/gmail",
        # "data/general/qq_browser",
        # "data/general/tencent_manager",
        # "data/general/tencent_video",
        # "data/game/huoying",
        # "data/mdnf",
        "data/general_EN/Facebook",
    ]
    for main_dir in app_scenarios:
        scene_dirs = [os.path.join(main_dir, scene) for scene in os.listdir(main_dir)
                      if os.path.isdir(os.path.join(main_dir, scene))]
        for record_dir in scene_dirs:
            print(f"Processing scene: {record_dir}")
            agent = RecordToSimpleStepsAgent(llm)
            task_infos = agent.execute(record_dir=record_dir, temperature=0)

            # pickle 格式的图
            graph_path = os.path.join(record_dir, "operation_ui_graph", "operation_graph.pkl")

            try:
                op_graph = OperationGraph.load(graph_path)  # 使用 pickle 加载图
                print(f"Loaded graph: |V|={len(op_graph.nodes)}, |E|={len(op_graph.edges)}")
            except Exception as e:
                print(f"Error loading graph from {graph_path}: {e}")


            # graph_path = os.path.join(record_dir, "operation_ui_graph", "operation_graph.json")
            # op_graph = OperationGraph.load(graph_path)
            # print(f"Loaded graph: |V|={len(op_graph.nodes)}, |E|={len(op_graph.edges)}")

            # print(json.dumps(task_infos, indent=2))



# 单场景运行
# def test_android_auto_exec_agent():
#     import cv2
#     import os
#     import json
#     import numpy as np
#     import logging
#     import adbutils
#
#     from LonghorizonAgent.common import llm_provider, utils
#     from LonghorizonAgent.agent.auto_execution_agent import AutoExecutionAgent, AutoExecutionConfig
#     from LonghorizonAgent.system.android_context import AndroidContext, AndroidContextConfig
#     from LonghorizonAgent.controller.android_controller import AndroidController
#     from LonghorizonAgent.prompts.auto_execution_prompt import AndroidExecSystemPrompt, AutoExecAgentPrompt
#     from LonghorizonAgent.common.llm_provider import LLMProvider
#
#     # model = "gemini-2.0-flash"
#     model = "gemini-2.5-pro-preview-05-06"
#     project = os.getenv("GOOGLE_PROJECT", "")
#     location = os.getenv("GOOGLE_LOCATION", "")
#     llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location)
#
#     devices = adbutils.adb.device_list()
#     android_device = devices[0].info["serialno"]
#     context_config = AndroidContextConfig(
#         device_id=android_device,
#         use_perception=True,
#         use_ocr=True,
#         use_ocr_rec=False,
#         perception_description_type="normal",
#         highlight_type="normal",
#         detect_split_x=2,
#         detect_split_y=2,
#         screenshot_save_dir=f"./tmp/agent_outputs/android_{android_device}"
#     )
#     system_context = AndroidContext(config=context_config)
#
#     controller = AndroidController(highlight_action=True)
#
#     agent_config = AutoExecutionConfig(
#         max_steps=100,
#         llm_temperature=0.4,
#         max_input_len=20,
#         keep_last_n_states=2,
#     )
#
#     agent = AutoExecutionAgent(
#         agent_config=agent_config,
#         controller=controller,
#         system_context=system_context,
#         llm=llm,
#         system_prompt_class=AndroidExecSystemPrompt,
#         agent_prompt_class=AutoExecAgentPrompt
#     )
#
#     # task = "1. 切换到后台。2. 依次滑动删除每个app的后台（注意是依次）。3. 返回桌面。"
#     # task_steps = ""
#     # task_infos = ""
#
# #     task = "使用NetCap启动地下城与勇士（dnf)，进行一系列操作后，返回NetCap点击暂停录制，最后后台杀掉dnf应用"
# #     task_steps = """
# # 1. 进入NetCap应用, 并开启悬浮窗
# # 2. 选择上报项目id: mdnf_sta
# # 3. 点击右上角开始录制, 自动进入dnf游戏
# # 4. 点击注销后，再选择qq登录
# # 5. 点击委托
# # 6. 点击特殊
# # 7. 选择马戏团第二幕
# # 8. 创建组队
# # 9. 退出组队
# # 10. 返回大厅
# # 11. 点击悬浮窗返回NetCap并点击右上角的停止录制
# # 12. 切到后台杀掉dnf应用
# # """
# #     task_infos = """
# # 1. 如果询问你是否继续之前的游戏内容，请点击取消
# # """
#
#     # task = "进入NetCap应用, 并开启悬浮窗"
#     task = "进入NetCap应用, 并开启悬浮窗, 然后点击右上角的开始录制。最后切换到后台杀掉所有的应用，返回首页"
#     task_steps = ""
#     task_infos = ""
#
#     task_infos_json = "data/1747014732.020043/task_infos.json"
#     with open(task_infos_json, "r", encoding='utf-8') as f:
#         task_infos = json.load(f)
#     task = task_infos["task_name"]
#     task_steps = ""
#     for i, task_step in enumerate(task_infos["task_steps"]):
#         task_steps += f"{i+1}. {task_step['action']}\n"
#     print(task)
#     print(task_steps)
#     agent_output_dir = agent.run(task, task_steps=task_steps, task_infos=task_infos)


# 多场景循环运行


def test_android_auto_exec_agent():
    import os
    import json
    from pathlib import Path  # 确保导入Path

    from LonghorizonAgent.common import llm_provider, utils
    from LonghorizonAgent.agent.auto_execution_agent import AutoExecutionAgent, AutoExecutionConfig
    from LonghorizonAgent.system.android_context import AndroidContext, AndroidContextConfig
    from LonghorizonAgent.controller.android_controller import AndroidController
    from LonghorizonAgent.prompts.auto_execution_prompt import AndroidExecSystemPrompt, AutoExecAgentPrompt
    from test_evalution import evaluate_action_performance
    from LonghorizonAgent.common.llm_provider import LLMProvider

    # model = "gemini-2.0-flash"
    model = "gemini-2.5-pro-preview-05-06"
    project = os.getenv("GOOGLE_PROJECT", "")
    location = os.getenv("GOOGLE_LOCATION", "")
    llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location)

    # # model = "GPT-4o"
    # AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    # MODEL = "gpt-4o"
    # llm = llm_provider.LLMProvider(llm_provider="azure_openai", model=MODEL, base_url=AZURE_OPENAI_ENDPOINT,
    #                                api_key=AZURE_OPENAI_API_KEY)

    # 移除adb连接
    # devices = adbutils.adb.device_list()
    # android_device = devices[0].info["serialno"]

    # 初始化变量
    initialization_file = './data/initialization.json'
    output_base_dir = Path("./tmp/agent_outputs")  # 修改输出目录

    scenario_metrics = {}  # 用于存储所有场景的评估指标

    # 遍历data
    data_root = Path("./data/mdnf_raw")
    for task_dir in data_root.iterdir():
        if not task_dir.is_dir():
            continue

        print(f"\n{'=' * 30}")
        print(f"开始执行任务目录: {task_dir.name}")
        print(f"{'=' * 30}")

        # 检查screenshot目录是否存在且非空
        screenshot_dir = task_dir / "screenshot"
        if not screenshot_dir.exists() or not any(screenshot_dir.glob("*.png")):
            print(f"跳过目录 {task_dir.name}：未找到截图或截图目录为空")
            continue

        # 任务文件路径
        task_infos_path = task_dir / "task_infos.json"
        if not task_infos_path.exists():
            print(f"跳过目录 {task_dir.name}：未找到task_infos.json")
            continue

        with open(task_infos_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)

        task_name = task_data.get("task_name", "未命名任务")
        task_steps = "\n".join(
            [f"{i + 1}. {step['action']}"
             for i, step in enumerate(task_data.get("task_steps", []))]
        )
        print(f"action prompt: {task_steps}")
        print(f"加载任务成功: {task_name}")

        task_output_dir = output_base_dir / task_dir.name

        task_output_dir.mkdir(parents=True, exist_ok=True)

        # ocr_results_path = task_output_dir / "ocr_results.json"

        context_config = AndroidContextConfig(
            local_screenshot_dir=str(screenshot_dir),
            use_perception=True,
            use_ocr=True,
            use_ocr_rec=False,
            perception_description_type="normal",
            highlight_grid_num=24,
            screenshot_save_dir=str(task_output_dir),  # 输出目录
            detect_split_x=2,
            detect_split_y=2,
            # ocr_results_path=str(ocr_results_path),  # OCR save patch
        )

        system_context = AndroidContext(config=context_config)

        controller = AndroidController(highlight_action=True)

        # 配置代理
        agent_config = AutoExecutionConfig(
            max_steps=100,
            llm_temperature=0.4,
            max_input_len=20,
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
            print(f"\nbegin 执行: {task_name}")
            print(f"截图来源: {screenshot_dir}")
            print(f"output: {task_output_dir}")

            if hasattr(system_context, 'reset_screenshot_index'):
                system_context.reset_screenshot_index()

            agent_output_dir = agent.run(
                task=task_name,
                task_steps=task_steps,
                task_infos=f"原始任务目录: {task_dir.name}"
            )

            print(f"任务完成，输出保存在: {agent_output_dir}")
        except Exception as e:
            import traceback
            print(f"任务执行异常: {str(e)}")
            traceback.print_exc()
            continue

        # 执行后立即进行评估
        try:
            task_name = task_dir.name
            predictions_path = Path(agent_output_dir) / "action_records.json"
            ground_truth_path = task_dir / "task_infos.json"

            if not predictions_path.exists():
                print(f"跳过评估 {task_name}: 预测文件不存在")
                continue
            if not ground_truth_path.exists():
                print(f"跳过评估 {task_name}: 真实标签文件不存在")
                continue

            # 调用评估函数
            metrics = evaluate_action_performance(
                predictions_path=str(predictions_path),
                ground_truth_path=str(ground_truth_path),
                output_dir=str(agent_output_dir)
            )

            if metrics:
                scenario_metrics[task_name] = metrics
                print(f"\n[场景评估结果] {task_name}:")
                print(f"  动作类型匹配准确率: {metrics['type_match_acc']}%")
                print(f"  动作精确匹配准确率: {metrics['exact_match_acc']}%")
                print(f"  点击动作匹配准确率: {metrics['click_match_acc']}%")
                print(f"  总步骤数: {metrics['total_steps']}")
        except Exception as e:
            print(f"评估任务 {task_name} 失败: {e}")

    # 所有场景完成后计算平均指标
    if scenario_metrics:
        print("\n\n[所有场景总体评估结果]")

        # 计算各个指标的加权平均值
        total_steps = sum(m['total_steps'] for m in scenario_metrics.values())
        weighted_type_match = sum(
            m['type_match_acc'] * m['total_steps'] for m in scenario_metrics.values()) / total_steps
        weighted_exact_match = sum(
            m['exact_match_acc'] * m['total_steps'] for m in scenario_metrics.values()) / total_steps

        # 计算点击动作准确率
        click_scenarios = [m for m in scenario_metrics.values() if 'click_match_acc' in m]
        avg_click_match = np.mean([m['click_match_acc'] for m in click_scenarios]) if click_scenarios else 0

        print(f"  平均动作类型匹配准确率: {weighted_type_match:.2f}% (加权平均值)")
        print(f"  平均动作精确匹配准确率: {weighted_exact_match:.2f}% (加权平均值)")
        print(f"  平均点击动作匹配准确率: {avg_click_match:.2f}% (简单平均值)")

        overall_path = output_base_dir / "overall_evaluation.json"
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump({
                "average_metrics": {
                    "weighted_type_match_acc": weighted_type_match,
                    "weighted_exact_match_acc": weighted_exact_match,
                    "simple_click_match_acc": avg_click_match
                },
                "scenario_metrics": scenario_metrics,
                "total_scenarios": len(scenario_metrics),
                "total_steps": total_steps
            }, f, indent=2, ensure_ascii=False)

        print(f"\n总体评估结果已保存至: {overall_path}")


def test_llm_with_perception():
    import adbutils
    from LonghorizonAgent.system.computer_context import ComputerContext, ComputerContextConfig
    from LonghorizonAgent.common import utils, llm_provider

    context_config = ComputerContextConfig(
        use_perception=True,
        use_ocr=True,
        use_ocr_rec=False,
        perception_description_type="md",
        highlight_type="normal",
        highlight_grid_num=24,
        screenshot_save_dir=f"./tmp/screenshots/computer",
        detect_split_x=2,
        detect_split_y=2,
    )
    context = ComputerContext(config=context_config)
    image_path = "data/1_20/1747014732.020043/screenshot/001.png"
    cur_state = context.update_state(image_path)

    highlight_screenshot_path = cur_state.highlight_screenshot_path

    model = "gemini-2.5-pro-preview-05-06"
    # model = "gemini-2.0-flash"
    google_key_json_path = "./assets/google_keys/turinglab-507d7c079329.json"
    project = os.environ.get("GOOGLE_PROJECT", "")
    location = os.environ.get("GOOGLE_LOCATION", "")
    llm = llm_provider.LLMProvider(llm_provider="gemini",
                                   model=model,
                                   project=project,
                                   location=location,
                                   google_key_json_path=google_key_json_path)

    system_prompt = """
    You are an expert GUI automation agent. Your goal is to answer the instructions specified by the user by interacting with a PC or mobile device GUI based on screenshots.

    You will be given a a screenshot which is highlighted of the current screen. UI elements like icons and text detected by a vision model are highlighted with semi-transparent colored boxes. Each box has an index number in its top-left corner.
    And you should output the highlighted index which is most relevant to the user's instructions.

    You MUST respond ONLY with a single valid JSON object in the following exact format. Do NOT include any text outside this JSON structure.

    ```json
    {
      "think": "Your reasoning goes here.",
      "index": {{highlighted_index}}
    }
    # Example Output
    ```json
    {
      "think": "Element center is within the highlighted box with index 3. Center is near the geometric center of the box.",
      "index": 3,
    }
    """
    user_prompt = "number 5"

    chat_messages = []
    chat_messages = llm.add_message("system", system_prompt, chat_messages)
    image_base64 = utils.encode_image(highlight_screenshot_path)
    chat_messages = llm.add_message("user", user_prompt, chat_messages, [image_base64])
    response_text = llm.invoke(chat_messages, temperature=0.)

    if not response_text:
        print("\nError: Received empty response from LLM.")
        return
    print(response_text)


if __name__ == '__main__':
    test_record_to_simple_steps()
    # test_android_auto_exec_agent()
    # test_llm_with_perception()
