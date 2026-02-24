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
            # 三个核心指标
            "type_match_acc": 0.0,
            "grounding_accuracy": 0.0,
            "step_success_rate": 0.0,
            # 各个操作指标
            "exact_match_acc": 0.0,
            "click_match_acc": 0.0,
            "long_press_match_acc": 0.0,
            "swipe_match_acc": 0.0,
            "input_match_acc": 0.0,
            "total_steps": 0,
            "action_type_stats": {}
        }

    type_match_count = 0  # Type分子：类型匹配的步骤数
    grounding_correct_count = 0  # GR分子：定位正确的步骤数
    step_success_count = 0  # SR分子：步骤成功的步骤数

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

    scene_success = True

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

            # 记录预测和真实动作类型
            step_result["gt_action"] = gt_action_type
            step_result["pred_action"] = pred_action

            if gt_action_type in action_type_stats:
                action_type_stats[gt_action_type]["count"] += 1

            # 类型匹配 (Type)
            type_match = (gt_action_type == pred_action)
            if type_match:
                type_match_count += 1

            # B. ground 准确度 (GR)
            grounding_ok = False

            if gt_action_type in ["click", "long_press", "input_text"]:
                # 点击类动作：检查点击point是否在bbox内
                if "resolved_coords" in pred and "pos" in pred["resolved_coords"] and gt_action_data.get("bbox"):
                    pred_coords = pred["resolved_coords"]["pos"]
                    grounding_ok = point_in_bbox(
                        pred_coords[0],
                        pred_coords[1],
                        gt_action_data["bbox"]
                    )

            elif gt_action_type in ["swipe", "drag"]:
                if ("resolved_coords" in pred and
                        "start" in pred["resolved_coords"] and
                        gt_action_data.get("bbox")):
                    pred_start = pred["resolved_coords"]["start"]
                    grounding_ok = point_in_bbox(
                        pred_start[0],
                        pred_start[1],
                        gt_action_data["bbox"]
                    )

            else:
                grounding_ok = True

            if grounding_ok:
                grounding_correct_count += 1

            if gt_action_type in ["click", "long_press", "input_text", "swipe", "drag"]:
                step_success = type_match and grounding_ok
            else:
                # 非坐标类动作：只需要type正确
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

    # 5. save
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
    elif action_str.startswith("Press"):
        return "press_key"
    else:
        return "unknown"


def extract_action_data(action_str: str, gt_step: dict) -> dict:
    """
    从动作字符串中提取动作数据
    :param action_str: 动作字符串
    :param gt_step: 真实标签步骤
    :return: 包含动作数据的字典
    """
    action_type = extract_action_type(action_str)
    result = {"type": action_type}

    if action_type == "click" or action_type == "long_press":
        coords = extract_coords_from_action(action_str)
        if coords:
            result["position"] = coords
        if "bbox" in gt_step["raw_action"]:
            result["bbox"] = gt_step["raw_action"]["bbox"]

    elif action_type == "swipe" or action_type == "drag":
        coords = extract_coords_from_action(action_str)
        if coords:
            result["start"], result["end"] = coords
        if "bbox" in gt_step["raw_action"]:
            result["bbox"] = gt_step["raw_action"]["bbox"]

    elif action_type == "input_text":
        coords = extract_coords_from_action(action_str)
        if coords:
            result["position"] = coords
        if "bbox" in gt_step["raw_action"]:
            result["bbox"] = gt_step["raw_action"]["bbox"]

    elif action_type == "press_key":
        if " " in action_str:
            key_name = action_str.split(maxsplit=1)[1]
            if key_name in ["home", "back", "recent", "power"]:
                result["key_name"] = key_name

    return result


def extract_coords_from_action(action_str: str) -> Optional[Tuple]:
    """
    从动作字符串中提取坐标
    :param action_str: 例如 "Tap (685, 867)", "Swipe (100,200) to (300,400)"
    :return: 坐标元组（单个点或起点终点）
    """
    try:
        # 尝试提取单个点坐标
        single_point_match = re.search(r'$(\d+),\s*(\d+)$', action_str)
        if single_point_match:
            return (float(single_point_match.group(1)), float(single_point_match.group(2)))

        # 尝试提取起点和终点坐标
        two_points_match = re.findall(r'$(\d+),\s*(\d+)$', action_str)
        if two_points_match and len(two_points_match) == 2:
            start = (float(two_points_match[0][0]), float(two_points_match[0][1]))
            end = (float(two_points_match[1][0]), float(two_points_match[1][1]))
            return start, end

    except Exception as e:
        print(f"提取坐标失败: {action_str}, 错误: {e}")

    return None


def point_in_bbox(x: float, y: float, bbox: list) -> bool:
    """
    检查点是否在边界框内（像素坐标）
    :param x: 点的x坐标（像素）
    :param y: 点的y坐标（像素）
    """
    if bbox is None or len(bbox) < 4:
        return False

    # 确保 bbox 坐标顺序正确（左上角到右下角）
    sorted_bbox = sorted([
        min(bbox[0], bbox[2]),
        min(bbox[1], bbox[3]),
        max(bbox[0], bbox[2]),
        max(bbox[1], bbox[3])
    ])[:4]  # 只取前4个元素以防多余值

    x1, y1, x2, y2 = sorted_bbox

    # 检查点是否在边界框内
    return (x1 <= x <= x2) and (y1 <= y <= y2)