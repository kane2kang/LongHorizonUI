import os
import shutil


def copy_task_and_screenshot(source_dir, target_dir):
    """
    递归复制所有游戏目录中的截图文件夹（screenshot）和任务信息文件

    Args:
        source_dir: 源目录 (data/game)
        target_dir: 目标目录 (data/game_new)
    """
    # 遍历源目录中的所有项
    for root, dirs, files in os.walk(source_dir):
        # 如果当前路径包含我们需要的文件结构
        if 'screenshot' in dirs or 'task_infos.json' in files:
            # 计算相对路径（相对于源目录）
            rel_path = os.path.relpath(root, source_dir)

            # 构建目标路径
            target_game_path = os.path.join(target_dir, rel_path)

            # 如果发现screenshot目录
            if 'screenshot' in dirs:
                src_screenshot = os.path.join(root, 'screenshot')
                dst_screenshot = os.path.join(target_game_path, 'screenshot')

                # 确保目标目录存在
                os.makedirs(os.path.dirname(dst_screenshot), exist_ok=True)

                # 复制截图目录
                if os.path.exists(dst_screenshot):
                    shutil.rmtree(dst_screenshot)
                shutil.copytree(src_screenshot, dst_screenshot)
                print(f"已复制截图目录: {src_screenshot} → {dst_screenshot}")

            # # 如果发现任务信息文件
            # if 'task_infos.json' in files:
            #     src_json = os.path.join(root, 'task_infos.json')
            #     dst_json = os.path.join(target_game_path, 'task_infos.json')
            #
            #     # 确保目标目录存在
            #     os.makedirs(os.path.dirname(dst_json), exist_ok=True)
            #
            #     # 复制JSON文件
            #     shutil.copy2(src_json, dst_json)
            #     print(f"已复制任务文件: {src_json} → {dst_json}")


if __name__ == "__main__":
    # 设置源目录和目标目录
    SOURCE_DIR = 'game'  # 源目录
    TARGET_DIR = 'game_new'  # 目标目录

    print(f"开始从 {SOURCE_DIR} 复制到 {TARGET_DIR}")
    copy_task_and_screenshot(SOURCE_DIR, TARGET_DIR)
    print("操作完成！已复制所有screenshot目录和任务文件")
