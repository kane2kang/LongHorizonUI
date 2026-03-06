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

    model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
    # model = "<your-model-variant>"
    project = os.getenv("LLM_PROJECT", "")
    location = os.getenv("LLM_LOCATION", "")
    llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location)

    # AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    # MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    # llm = llm_provider.LLMProvider(llm_provider="azure_openai", model=MODEL, base_url=AZURE_OPENAI_ENDPOINT,
    #                                api_key=AZURE_OPENAI_API_KEY)
    main_dir = "data/sample_raw"
    # 多应用场景
    # app_scenarios = [
    #     "data/general/app_a",
    #     "data/general/app_b",
    #     "data/general/app_c",
    #     "data/general/app_d"
    #     "data/general/app_e"
    #     # "data/general/gmail",
    # ]
    # for main_dir in app_scenarios:
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
    genreal_scenarios = {
        "app_a": "data/general/app_a",
        "app_b": "data/general/app_b",
        "app_c": "data/general/app_c",
        "app_d": "data/general/app_d",
        "app_e": "data/general/app_e",
        "app_f": "data/general/app_f",
        "app_g": "data/general/app_g",
        "app_h": "data/general/app_h",
        "app_i": "data/general/app_i",
        "app_j": "data/general/app_j"
    }

    model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
    project = os.getenv("LLM_PROJECT", "")
    location = os.getenv("LLM_LOCATION", "")
    llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location)

    # # model = "<your-model>"
    # AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    # MODEL = os.getenv("LLM_MODEL", "gpt-4o")
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
    for scenario_name, scenario_path in genreal_scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"开始处理场景: {scenario_name}")
        print(f"{'=' * 60}")

        data_root = Path(scenario_path)
        output_base_dir = Path("./tmp/agent_outputs") / scenario_name
        scenario_metrics = {}

        # 遍历当前场景下的所有任务
        for task_dir in data_root.iterdir():
            if not task_dir.is_dir():
                continue

            print(f"\n{'=' * 30}")
            print(f"开始执行任务目录: {task_dir.name}")
            print(f"{'=' * 30}")

            screenshot_dir = task_dir / "screenshot"
            if not screenshot_dir.exists() or not any(screenshot_dir.glob("*.png")):
                print(f"跳过目录 {task_dir.name}：未找到截图或截图目录为空")
                continue

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

            # 执行后evalution
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

                    # 1. 输出三核心指标
                    print(f"  type_match_acc(Type): {metrics['type_match_acc']}%")
                    print(f"  grounding_accuracy(GR): {metrics['grounding_accuracy']}%")
                    print(f"  step_success_rate(SR): {metrics['step_success_rate']}%")
                    print(f"  total_steps: {metrics['total_steps']}")

                    # 2. 各动作类型统计
                    action_type_stats = metrics.get("action_type_stats", {})
                    for action_type, stats in action_type_stats.items():
                        if stats["count"] > 0:
                            accuracy = stats.get("accuracy", 0.0)
                            print(f"  - {action_type}动作准确率: {accuracy}% ({stats['exact_match']}/{stats['count']})")

                    # 3. 动作类型分布统计
                    # 计算各动作类型占比
                    action_stats = metrics.get("action_type_stats", {})
                    total = metrics['total_steps']
                    distribution = {type: f"{(stats['count'] / total) * 100:.1f}%"
                                    for type, stats in action_stats.items() if stats['count'] > 0}

                    if distribution:
                        print(f"\n  动作类型分布:")
                        for action_type, percent in distribution.items():
                            print(f"    - {action_type}: {percent}")


                    if "click_match_acc" in metrics:
                        print(f"  click动作精确匹配率: {metrics['click_match_acc']}%")
                    if "long_press_match_acc" in metrics:
                        print(f"  long_press动作精确匹配率: {metrics['long_press_match_acc']}%")
                    if "swipe_match_acc" in metrics:
                        print(f"  swipe动作精确匹配率: {metrics['swipe_match_acc']}%")
                    if "input_match_acc" in metrics:
                        print(f"  input动作精确匹配率: {metrics['input_match_acc']}%")

            except Exception as e:
                print(f"评估任务 {task_name} 失败: {e}")

        # 所有场景完成后计算平均指标
        if scenario_metrics:
            print(f"\n\n[{scenario_name}场景总体评估结果]")

            total_steps = sum(m['total_steps'] for m in scenario_metrics.values())

            # 计算三核心指标的加权平均值
            weighted_type_match = sum(
                m['type_match_acc'] * m['total_steps'] for m in scenario_metrics.values()
            ) / total_steps

            weighted_grounding = sum(
                m['grounding_accuracy'] * m['total_steps'] for m in scenario_metrics.values()
            ) / total_steps

            weighted_sr = sum(
                m['step_success_rate'] * m['total_steps'] for m in scenario_metrics.values()
            ) / total_steps

            # 计算ESR
            passed_scenes_count = sum(1 for m in scenario_metrics.values() if m['scene_success'])
            esr_rate = passed_scenes_count / len(scenario_metrics) * 100

            # 打印三核心指标
            print(f"  weighted_type_match(Type): {weighted_type_match:.2f}% ")
            print(f"  weighted_grounding(GR): {weighted_grounding:.2f}% ")
            print(f"  weighted_SR: {weighted_sr:.2f}% ")
            print(f"  scenario_metrics: {len(scenario_metrics)}")
            print(f"  total_steps: {total_steps}")

            global_action_stats = {
                "click": {"exact_match": 0, "count": 0},
                "long_press": {"exact_match": 0, "count": 0},
                "swipe": {"exact_match": 0, "count": 0},
                "drag": {"exact_match": 0, "count": 0},
                "input_text": {"exact_match": 0, "count": 0},
                "press_key": {"exact_match": 0, "count": 0}
            }

            for metrics in scenario_metrics.values():
                action_type_stats = metrics.get("action_type_stats", {})
                for action_type, stats in action_type_stats.items():
                    if action_type in global_action_stats:
                        global_action_stats[action_type]["exact_match"] += stats.get("exact_match", 0)
                        global_action_stats[action_type]["count"] += stats.get("count", 0)

            # 计算所有动作类型的全局准确率
            global_accuracies = {}
            for action_type, stats in global_action_stats.items():
                count = stats["count"]
                exact_match = stats["exact_match"]
                if count > 0:
                    accuracy = round((exact_match / count) * 100, 2)
                    global_accuracies[action_type] = accuracy
                else:
                    global_accuracies[action_type] = 0.0

            print(f"\n  动作类型详细统计:")
            for action_type, accuracy in global_accuracies.items():
                if global_action_stats[action_type]["count"] > 0:
                    print(f"    {action_type}动作精确匹配率: {accuracy}% "
                          f"({global_action_stats[action_type]['exact_match']}/{global_action_stats[action_type]['count']})")

            # 保存总体评估结果
            # 保存场景总体评估结果（按场景名称）
            overall_path = output_base_dir / f"{scenario_name}_overall_evaluation.json"
            with open(overall_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "average_metrics": {
                        "weighted_type_match_acc": weighted_type_match,
                        "weighted_grounding_accuracy": weighted_grounding,
                        "weighted_step_success_rate": weighted_sr,
                        "esr_rate": esr_rate,  # ESR指标
                    },
                    "esr_passed_count": passed_scenes_count,
                    "esr_total_count": len(scenario_metrics),
                    "global_action_accuracies": global_accuracies,
                    "global_action_stats": global_action_stats,
                    "scenario_metrics": scenario_metrics,
                    "total_scenarios": len(scenario_metrics),
                    "total_steps": total_steps
                }, f, indent=2, ensure_ascii=False)

            print(f"\n{scenario_name}场景总体评估结果已保存至: {overall_path}")


if __name__ == '__main__':
    # test_record_to_simple_steps()
    test_android_auto_exec_agent()
