import os
import json
from collections import Counter


def analyze_action_types(base_dir):
    """
    分析指定目录下所有场景文件夹中task_infos.json文件里的动作类型

    Args:
        base_dir (str): 包含所有场景文件夹的根目录路径
    """
    # 初始化计数器
    action_counter = Counter()

    # 遍历基础目录下的所有子目录
    for root, dirs, files in os.walk(base_dir):
        # 检查当前目录中是否有task_infos.json文件
        if "task_infos.json" in files:
            json_path = os.path.join(root, "task_infos.json")

            try:
                # 读取JSON文件
                with open(json_path, "r", encoding="utf-8") as f:
                    task_info = json.load(f)

                # 输出调试信息，查看是否读取成功
                print(f"成功读取文件: {json_path}")

                # 解析所有步骤中的动作类型
                for step in task_info.get("task_steps", []):
                    if "raw_action" in step and "action" in step["raw_action"]:
                        # 提取动作类型（动作字符串的第一个单词）
                        action_str = step["raw_action"]["action"]
                        action_type = action_str.split()[0]  # 获取第一个单词
                        action_counter[action_type] += 1
            except Exception as e:
                print(f"解析文件失败: {json_path} - {str(e)}")

    # 输出统计结果
    print(f"发现 {len(action_counter)} 种不同的动作类型")
    print("\n动作类型统计:")
    for action, count in action_counter.most_common():
        print(f"{action}: {count}次")

    return action_counter

if __name__ == "__main__":
    # 定义要分析的文件夹列表
    app_scenarios = [
        # 通用应用
        "general/app_a",
        "general/app_b",
        "general/app_c",
        "general/app_d",
        "general/app_e",
        "general/app_f",
        "general/app_g",
        "general/app_h",
        "general/app_i",
        "general/app_j",
        "general/app_k",
        # 游戏应用
        # "data/game/game_a",
        # "data/game/game_b",
        # "data/game/game_c",
        # "data/game/game_d",
        # "data/game/game_e"
    ]

    # 循环处理多个目录
    for base_directory in app_scenarios:
        print(f"\n分析目录: {base_directory}")

        # 确保目录存在
        if not os.path.exists(base_directory):
            print(f"错误: 指定的目录不存在 - {base_directory}")
        else:
            # 执行分析
            action_stats = analyze_action_types(base_directory)

            # 额外信息：生成动作类型列表
            print("\n动作类型列表:")
            print(", ".join(sorted(action_stats.keys())))
