import json
import os
from collections import OrderedDict


def convert_json_structure(source_data):
    """将源JSON结构转换为目标格式"""
    target_data = OrderedDict()
    target_data["task_name"] = source_data["goal"]
    target_data["task_steps"] = []

    # 使用源数据中的截图数量来计算时间戳增量
    num_screenshots = len(source_data["screenshots"])
    base_timestamp = 1747014870.0
    timestamp_step = 1.0 / max(1, num_screenshots)  # 防止除以零

    for step_idx, action_data in enumerate(source_data["actions"]):
        step = OrderedDict()
        step_id = f"{step_idx + 1:03d}"

        # 获取当前步骤的指令
        if step_idx < len(source_data["step_instructions"]):
            action_desc = source_data["step_instructions"][step_idx]
        else:
            # 如果指令数量少于动作数量，生成默认描述
            action_type = action_data.get("action_type", "unknown")
            action_desc = f"执行{action_type}操作"

        # 设置动作描述和结果（源数据无执行结果信息）
        step["action"] = action_desc
        step["action_result"] = "执行结果已记录在截图中"
        step["step_id"] = step_id

        # 构建原始动作数据
        raw_action = OrderedDict()

        # 时间戳生成（每个步骤递增）
        current_timestamp = base_timestamp + step_idx * timestamp_step
        raw_action["timestamp"] = round(current_timestamp, 6)

        # 处理不同类型的动作
        action_type = action_data["action_type"]
        if action_type == "click":
            raw_action["action"] = f"Tap ({action_data['x']}, {action_data['y']})"
        elif action_type == "open_app":
            raw_action["action"] = f"OpenApp: {action_data['app_name']}"
        elif action_type == "wait":
            raw_action["action"] = "Wait"
        else:
            raw_action["action"] = f"{action_type} (未识别操作)"

        # 获取当前步骤对应的截图索引（跳过初始状态截图）
        screenshot_idx = min(step_idx + 1, len(source_data["screenshots"]) - 1)

        # 处理截图路径 - 只保留文件名
        orig_screenshot_path = source_data["screenshots"][screenshot_idx]
        screenshot_filename = os.path.basename(orig_screenshot_path)
        raw_action["screenshot"] = os.path.join("screenshot", screenshot_filename)

        # 添加截图尺寸
        raw_action["screenshot_width"] = source_data["widths"][screenshot_idx]
        raw_action["screenshot_height"] = source_data["heights"][screenshot_idx]

        step["raw_action"] = raw_action
        target_data["task_steps"].append(step)

    return target_data


def main():
    # 输入文件路径
    source_path = "android_control/episode_0_result.json"

    # 输出文件路径
    target_path = os.path.join(os.path.dirname(source_path), "converted_episode_0_result.json")

    try:
        # 读取源JSON文件
        with open(source_path, "r", encoding="utf-8") as f:
            source_data = json.load(f)

        # 转换为目标格式
        converted_data = convert_json_structure(source_data)

        # 保存转换后的JSON
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

        print(f"转换成功! 结果已保存至: {target_path}")

    except FileNotFoundError:
        print(f"错误: 找不到源文件 {source_path}")
    except json.JSONDecodeError:
        print(f"错误: 文件 {source_path} 不是有效的JSON格式")
    except KeyError as e:
        print(f"错误: JSON文件中缺少必要字段 {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()