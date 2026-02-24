


import json
import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


def evaluate_action_performance(predictions_path: str,
                                ground_truth_path: str,
                                output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    评估所有类型动作的执行性能
    :param predictions_path: 模型预测结果文件路径 (action_records.json)
    :param ground_truth_path: 真实标签文件路径 (task_infos.json)
    :param output_dir: 评估结果保存目录
    :return: 包含评估指标的字典
    """
    # 1. 加载预测数据和真实标签
    try:
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
            gt_steps = ground_truth.get("task_steps", [])
    except Exception as e:
        print(f"加载文件失败: {e}")
        return {
            "error": f"文件加载失败: {e}",
            # 三核心指标
            "type_match_acc": 0.0,
            "grounding_accuracy": 0.0,
            "step_success_rate": 0.0,
            # 原有指标
            "exact_match_acc": 0.0,
            "click_match_acc": 0.0,
            "long_press_match_acc": 0.0,
            "swipe_match_acc": 0.0,
            "input_match_acc": 0.0,
            "total_steps": 0,
            "action_type_stats": {}
        }

    # 2. 初始化统计变量（三核心指标）
    type_match_count = 0  # Type分子：类型匹配的步骤数
    grounding_correct_count = 0  # GR分子：定位正确的步骤数
    step_success_count = 0  # SR分子：步骤成功的步骤数

    # 原有指标统计变量
    exact_match_count = 0
    click_match_count = 0
    long_press_match_count = 0
    swipe_match_count = 0
    input_match_count = 0
    press_key_match_count = 0
    drag_match_count = 0

    total_actions = min(len(predictions), len(gt_steps))
    results = []
    action_type_stats = {
        "click": {"count": 0, "exact_match": 0},
        "long_press": {"count": 0, "exact_match": 0},
        "swipe": {"count": 0, "exact_match": 0},
        "input_text": {"count": 0, "exact_match": 0},
        "press_key": {"count": 0, "exact_match": 0},
        "drag": {"count": 0, "exact_match": 0}
    }

    scene_success = True  # 初始假设整个场景成功

    # 3. 处理每个步骤
    for i in range(total_actions):
        step_result = {"step": i + 1}

        pred = predictions[i]
        pred_action = pred["action_type"]  # 'click', 'long_press', 'swipe', 'input_text', etc.

        try:
            gt_step = gt_steps[i]
            gt_action_str = gt_step["raw_action"]["action"]
            gt_action_type = extract_action_type(gt_action_str)
            gt_action_data = extract_action_data(gt_action_str, gt_step)

            screen_width = gt_step["raw_action"].get("screenshot_width", 1080)
            screen_height = gt_step["raw_action"].get("screenshot_height", 2312)

            # 记录预测和真实动作类型
            step_result["gt_action"] = gt_action_type
            step_result["pred_action"] = pred_action

            if gt_action_type in action_type_stats:
                action_type_stats[gt_action_type]["count"] += 1

            # A. 类型匹配检查 (Type指标分子)
            type_match = (gt_action_type == pred_action)
            if type_match:
                type_match_count += 1

                # B. 定位检查 (GR指标分子)
                grounding_ok = False

                # 根据动作类型进行不同的定位检查
                if gt_action_type in ["click", "long_press", "input_text"]:
                    if "resolved_coords" in pred and "pos" in pred["resolved_coords"] and gt_action_data.get("bbox"):
                        pred_coords = pred["resolved_coords"]["pos"]
                        # 移除不必要的参数传递
                        grounding_ok = point_in_bbox(
                            pred_coords[0],
                            pred_coords[1],
                            gt_action_data["bbox"]
                        )

                elif gt_action_type in ["swipe", "drag"]:
                    if ("resolved_coords" in pred and
                            "start" in pred["resolved_coords"] and
                            "end" in pred["resolved_coords"] and
                            gt_action_data.get("start_bbox") and
                            gt_action_data.get("end_bbox")):
                        # 检查起点和终点都在相应框内
                        start_ok = point_in_bbox(
                            pred["resolved_coords"]["start"][0],
                            pred["resolved_coords"]["start"][1],
                            gt_action_data["start_bbox"]
                        )
                        end_ok = point_in_bbox(
                            pred["resolved_coords"]["end"][0],
                            pred["resolved_coords"]["end"][1],
                            gt_action_data["end_bbox"]
                        )
                        grounding_ok = start_ok and end_ok

            else:
                grounding_ok = True

            if grounding_ok:
                grounding_correct_count += 1

            if gt_action_type in ["click", "long_press", "input_text", "swipe", "drag"]:
                # 坐标类动作：需要类型和定位都正确
                step_success = type_match and grounding_ok
            else:
                # 非坐标类动作：只需要类型正确
                step_success = type_match

            if step_success:
                step_success_count += 1

            exact_match = False
            if gt_action_type == "click" or gt_action_type == "long_press":
                if "resolved_coords" in pred and "pos" in pred["resolved_coords"] and gt_action_data.get("position"):
                    pred_coords = pred["resolved_coords"]["pos"]
                    exact_match = point_in_bbox(pred_coords[0], pred_coords[1], gt_action_data["position"])

                if gt_action_type == "click" and exact_match:
                    click_match_count += 1

                if gt_action_type == "long_press" and exact_match:
                    long_press_match_count += 1

            elif gt_action_type == "swipe" or gt_action_type == "drag":
                if ("resolved_coords" in pred and
                        "start" in pred["resolved_coords"] and "end" in pred["resolved_coords"] and
                        gt_action_data.get("start") and gt_action_data.get("end")):
                    pred_start = pred["resolved_coords"]["start"]
                    pred_end = pred["resolved_coords"]["end"]

                    start_match = point_in_bbox(pred_start[0], pred_start[1], gt_action_data["start"])
                    end_match = point_in_bbox(pred_end[0], pred_end[1], gt_action_data["end"])

                    exact_match = start_match and end_match

                if gt_action_type == "swipe" and exact_match:
                    swipe_match_count += 1

                if gt_action_type == "drag" and exact_match:
                    drag_match_count += 1

            elif gt_action_type == "input_text":
                if "resolved_coords" in pred and "pos" in pred["resolved_coords"] and gt_action_data.get("position"):
                    pred_coords = pred["resolved_coords"]["pos"]
                    exact_match = point_in_bbox(pred_coords[0], pred_coords[1], gt_action_data["position"])

                if exact_match:
                    input_match_count += 1

            elif gt_action_type == "press_key":
                if pred_action == "press_key" and "action_params" in pred:
                    key_param = pred["action_params"]
                    exact_match = (key_param.get("key_name") == gt_action_data.get("key_name"))

                if exact_match:
                    press_key_match_count += 1

            step_result["exact_match"] = exact_match
            if exact_match:
                exact_match_count += 1
                if gt_action_type in action_type_stats:
                    action_type_stats[gt_action_type]["exact_match"] += 1

            step_result["type_match"] = type_match
            step_result["grounding_ok"] = grounding_ok
            step_result["step_success"] = step_success

            if not step_success:
                scene_success = False

        except Exception as e:
            print(f"处理步骤 {i + 1} 时出错: {e}")
            step_result["error"] = str(e)
            step_result["type_match"] = False
            step_result["grounding_ok"] = False
            step_result["step_success"] = False
            step_result["exact_match"] = False
            scene_success = False

        results.append(step_result)

    metrics = {
        # 三核心指标
        "type_match_acc": round(type_match_count / total_actions * 100, 2) if total_actions > 0 else 0.0,
        "grounding_accuracy": round(grounding_correct_count / total_actions * 100, 2) if total_actions > 0 else 0.0,
        "step_success_rate": round(step_success_count / total_actions * 100, 2) if total_actions > 0 else 0.0,

        # 原有指标
        "exact_match_acc": round(exact_match_count / total_actions * 100, 2) if total_actions > 0 else 0.0,
        "click_match_acc": round(click_match_count / max(action_type_stats["click"]["count"], 1) * 100, 2) if
        action_type_stats["click"]["count"] > 0 else 0.0,
        "long_press_match_acc": round(long_press_match_count / max(action_type_stats["long_press"]["count"], 1) * 100,
                                      2) if action_type_stats["long_press"]["count"] > 0 else 0.0,
        "swipe_match_acc": round(swipe_match_count / max(action_type_stats["swipe"]["count"], 1) * 100, 2) if
        action_type_stats["swipe"]["count"] > 0 else 0.0,
        "input_match_acc": round(input_match_count / max(action_type_stats["input_text"]["count"], 1) * 100, 2) if
        action_type_stats["input_text"]["count"] > 0 else 0.0,
        "press_key_match_acc": round(press_key_match_count / max(action_type_stats["press_key"]["count"], 1) * 100,
                                     2) if action_type_stats["press_key"]["count"] > 0 else 0.0,
        "drag_match_acc": round(drag_match_count / max(action_type_stats["drag"]["count"], 1) * 100, 2) if
        action_type_stats["drag"]["count"] > 0 else 0.0,
        "total_steps": total_actions,
        "action_type_stats": action_type_stats,
        "scene_success": scene_success,  # 新增ESR指标
    }

    # 为每个动作类型计算精度
    for action_type, stats in action_type_stats.items():
        stats["accuracy"] = round(stats["exact_match"] / max(stats["count"], 1) * 100, 2) if stats["count"] > 0 else 0.0

    # 5. 保存评估结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        eval_file = os.path.join(output_dir, "evaluation_results.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "detailed_results": results,
                "predictions_path": predictions_path,
                "ground_truth_path": ground_truth_path
            }, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存至: {eval_file}")

    return metrics


def extract_action_type(action_str: str) -> str:
    """
    从动作字符串中提取动作类型
    :param action_str: 例如 "Tap (685, 867)", "Swipe (100,200) to (300,400)"
    :return: 动作类型 ("click", "long_press", "swipe", etc.)
    """
    if action_str.startswith("Tap"):
        return "click"
    elif action_str.startswith("LongPress") or action_str.startswith("Long Press"):
        return "long_press"
    elif action_str.startswith("Swipe"):
        return "swipe"
    elif action_str.startswith("Drag"):
        return "drag"
    elif action_str.startswith("Input"):
        return "input_text"
    elif action_str.startswith("View"):
        return "press_key"
    else:
        return "unknown"


def point_in_bbox(x: float, y: float, bbox: list) -> bool:
    """简化参数，专注坐标检查"""
    if not bbox or len(bbox) < 4:
        return False

    # 确保bbox坐标有序
    x1, y1, x2, y2 = sorted_bbox(bbox)
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def extract_action_data(action_str: str, gt_step: dict) -> dict:
    """完善坐标提取逻辑，尤其是swipe/drag动作"""
    action_type = extract_action_type(action_str)
    result = {"type": action_type}

    # 处理有点击坐标的动作
    if action_type in ["click", "long_press", "input_text"]:
        if coords := extract_coords_from_action(action_str):
            result["position"] = coords
        if "bbox" in gt_step["raw_action"]:
            result["bbox"] = sorted_bbox(gt_step["raw_action"]["bbox"])

    # 完善swipe/drag动作处理
    elif action_type in ["swipe", "drag"]:
        if coords := extract_coords_from_action(action_str):
            start, end = coords

            # 分别处理起点和终点的bbox
            if "start_bbox" in gt_step["raw_action"]:
                result["start_bbox"] = sorted_bbox(gt_step["raw_action"]["start_bbox"])
            if "end_bbox" in gt_step["raw_action"]:
                result["end_bbox"] = sorted_bbox(gt_step["raw_action"]["end_bbox"])

    # 处理文本输入的内容验证
    elif action_type == "input_text":
        result["text"] = gt_step["raw_action"].get("input_text", "")

    return result

def sorted_bbox(bbox: list) -> list:
    """确保bbox坐标正确排序"""
    return [
        min(bbox[0], bbox[2]),
        min(bbox[1], bbox[3]),
        max(bbox[0], bbox[2]),
        max(bbox[1], bbox[3])
    ]


def extract_coords_from_action(action_str: str) -> tuple:
    """
    从动作字符串中提取坐标
    :param action_str: 例如 "Tap (685, 867)", "Swipe (100,200) to (300,400)"
    :return: 坐标元组（单个点或起点终点）
    """
    try:
        if "(" in action_str and ")" in action_str and " to " not in action_str:
            start_idx = action_str.find("(") + 1
            end_idx = action_str.find(")")
            coords_str = action_str[start_idx:end_idx].strip()
            x, y = map(float, coords_str.split(","))
            return (x, y)

        elif " to " in action_str:
            parts = action_str.split(" to ")
            if len(parts) == 2:
                start_str = parts[0].strip()
                end_str = parts[1].strip()
                start_idx = start_str.find("(") + 1
                end_idx = start_str.find(")")
                coords_str = start_str[start_idx:end_idx].strip()
                start_x, start_y = map(float, coords_str.split(","))
                start_idx = end_str.find("(") + 1
                end_idx = end_str.find(")")
                coords_str = end_str[start_idx:end_idx].strip()
                end_x, end_y = map(float, coords_str.split(","))

                return (start_x, start_y), (end_x, end_y)

    except Exception as e:
        print(f"提取坐标失败: {action_str}, 错误: {e}")

    return None

