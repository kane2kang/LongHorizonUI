import json
import os
from pathlib import Path
import numpy as np


def evaluate_action_performance(predictions_path, ground_truth_path, output_dir=None):
    """
    评估动作执行性能
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
        return None

    # 2. 初始化统计变量
    type_match_count = 0
    exact_match_count = 0
    click_match_count = 0
    total_steps = min(len(predictions), len(gt_steps))
    gt_click_steps = 0
    results = []

    # 3. 处理每个步骤
    for i in range(total_steps):
        step_result = {"step": i + 1}

        pred = predictions[i]
        pred_action = pred["action_type"]  # 'click', 'long_press', 'swipe', etc.
        pred_coords = pred["resolved_coords"]["pos"] if "pos" in pred["resolved_coords"] else None

        # 获取gt
        gt_step = gt_steps[i]
        gt_action = "click"  # 真实标签中所有动作都是Tap
        gt_action_str = gt_step["raw_action"]["action"]
        gt_coords = extract_coords_from_action(gt_action_str)
        gt_bbox = gt_step["raw_action"].get("bbox")

        step_result["gt_action"] = gt_action
        step_result["pred_action"] = pred_action

        type_match = (gt_action == pred_action)
        step_result["type_match"] = type_match
        if type_match:
            type_match_count += 1

        exact_match = False
        if type_match and gt_bbox and pred_coords:
            exact_match = point_in_bbox(pred_coords[0], pred_coords[1], gt_bbox)
            step_result["exact_match"] = exact_match
            if exact_match:
                exact_match_count += 1

            # 统计点击动作的匹配
            if gt_action == "click":
                gt_click_steps += 1
                if exact_match:
                    click_match_count += 1
        else:
            step_result["exact_match"] = False

        results.append(step_result)

    # 4. 计算评估指标
    metrics = {
        "type_match_acc": round(type_match_count / total_steps * 100, 2) if total_steps > 0 else 0,
        "exact_match_acc": round(exact_match_count / total_steps * 100, 2) if total_steps > 0 else 0,
        "click_match_acc": round(click_match_count / gt_click_steps * 100, 2) if gt_click_steps > 0 else 0,
        "total_steps": total_steps
    }

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


def extract_coords_from_action(action_str):
    """
    从动作字符串中提取坐标
    :param action_str: 例如 "Tap (685, 867)"
    :return: (x, y) 坐标元组
    """
    try:
        # 匹配括号内的坐标
        start_idx = action_str.find("(") + 1
        end_idx = action_str.find(")")
        coords_str = action_str[start_idx:end_idx].strip()
        x, y = map(float, coords_str.split(","))
        return x, y
    except Exception as e:
        print(f"提取坐标失败: {action_str}, 错误: {e}")
        return None


def point_in_bbox(x, y, bbox):
    """
    检查点是否在边界框内
    :param x: 点的x坐标
    :param y: 点的y坐标
    :param bbox: 边界框 [x1, y1, x2, y2]
    :return: 布尔值，表示点是否在框内
    """
    if len(bbox) < 4:
        return False

    x1, y1, x2, y2 = bbox[:4]
    return (x1 <= x <= x2) and (y1 <= y <= y2)