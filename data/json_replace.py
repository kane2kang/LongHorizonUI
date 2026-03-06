# import os
# import shutil
#
# # 配置路径
# base_dir_old = 'general'
# base_dir_new = 'general_new'
#
# # 遍历general_new目录
# for root, dirs, files in os.walk(base_dir_new):
#     for file_name in files:
#         # 仅处理task_infos.json文件
#         if file_name == 'task_infos.json':
#             # 新文件的完整路径
#             new_file_path = os.path.join(root, file_name)
#
#             # 计算旧文件对应路径
#             relative_path = os.path.relpath(root, base_dir_new)
#             old_file_dir = os.path.join(base_dir_old, relative_path)
#             old_file_path = os.path.join(old_file_dir, file_name)
#
#             # 确保旧目录存在
#             os.makedirs(old_file_dir, exist_ok=True)
#
#             # 复制并替换文件
#             shutil.copy2(new_file_path, old_file_path)
#             print(f"已更新: {old_file_path}")
#
# print("\n操作完成！所有修正文件已同步到 data/general")


import os
import shutil
import json


def replace_corrected_files(original_root="game_new", corrected_root="game_new/Analysis"):
    """
    使用修正后的文件替换原始测试文件

    :param original_root: 原始文件根目录
    :param corrected_root: 修正文件根目录
    """
    # 1. 遍历所有修正文件
    for game in os.listdir(corrected_root):
        game_corrected_dir = os.path.join(corrected_root, game)
        if not os.path.isdir(game_corrected_dir):
            continue

        print(f"处理游戏: {game}")
        # 2. 获取所有修正文件
        for corrected_file in os.listdir(game_corrected_dir):
            if not corrected_file.endswith(".json"):
                continue

            # 3. 提取时间戳标识
            timestamp = corrected_file.rsplit(".json", 1)[0]

            # 4. 定位原始文件路径
            original_dir = os.path.join(original_root, game, timestamp)
            original_file = os.path.join(original_dir, "task_infos.json")
            corrected_file_path = os.path.join(game_corrected_dir, corrected_file)

            # 5. 创建备份并替换文件
            if os.path.exists(original_file):
                try:
                    # 创建备份目录
                    backup_dir = os.path.join(original_dir, "backup")
                    os.makedirs(backup_dir, exist_ok=True)

                    # 备份原始文件
                    backup_file = os.path.join(backup_dir, f"task_infos_backup.json")
                    shutil.copy2(original_file, backup_file)
                    print(f"  → 已备份: {backup_file}")

                    # 替换文件
                    shutil.copy2(corrected_file_path, original_file)
                    print(f"  ✓ 已替换: {original_file}")

                    # 验证文件完整性
                    if validate_replacement(original_file, corrected_file_path):
                        print(f"  ✓ 验证通过: 文件替换成功")
                    else:
                        print(f"  ⚠ 验证失败: 文件内容不一致")
                        # 恢复备份
                        shutil.copy2(backup_file, original_file)
                        print(f"  → 已恢复备份")

                except Exception as e:
                    print(f"  ✗ 处理失败({timestamp}): {str(e)}")
            else:
                print(f"  ⚠ 原始文件不存在: {original_file}")


def validate_replacement(original_path, corrected_path):
    """验证文件替换是否成功（比较文件内容）"""
    try:
        with open(original_path, 'r') as f1, open(corrected_path, 'r') as f2:
            original_content = json.load(f1)
            corrected_content = json.load(f2)
            return original_content == corrected_content
    except:
        # 如果JSON解析失败，使用二进制比较
        with open(original_path, 'rb') as f1, open(corrected_path, 'rb') as f2:
            return f1.read() == f2.read()


if __name__ == "__main__":
    print("开始替换修正文件...")
    replace_corrected_files()
    print("文件替换操作完成！")