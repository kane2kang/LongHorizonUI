# import os
# import shutil
#
# # 设置根目录路径
# root_dir = r"mdnf_raw"
#
# # 遍历根目录下的所有子目录（即每个场景文件夹）
# for scene_name in os.listdir(root_dir):
#     scene_path = os.path.join(root_dir, scene_name)
#     if os.path.isdir(scene_path):  # 确保是文件夹
#         target_folder = os.path.join(scene_path, "operation_ui_graph")
#         if os.path.exists(target_folder) and os.path.isdir(target_folder):
#             print(f"正在删除: {target_folder}")
#             shutil.rmtree(target_folder)
# import os
# import shutil
#
# # 要处理的多个根目录列表
# base_paths = [
#         # "general_new/qq",
#         # "general_new/qq_music",
#         # "general_new/tencent_meeting",
#         # "general_new/weixin_new",
#         # "general_new/qq_browser",
#         # "general_new/tencent_manager",
#         # "general_new/tencent_video",
#         # "general_new/tencent_document",
#         # "general_new/weishi"
#    "game_new/DNF",
#     "game_new/hero",
#     "game_new/honor_kings",
#     "game_new/huoying",
#     "game_new/jinchanchan",
#     "game_new/League_of_Legends",
#     "game_new/Peaceful_Elite",
#     "game_new/QQFlyCar",
#     "game_new/sanjiaozhou",
#     "game_new/yuanmeng"
# ]
#
# # 遍历根目录列表
# for base_path in base_paths:
#     # 检查每个根目录是否存在
#     if os.path.exists(base_path):
#         # 遍历当前根目录下的所有子文件夹
#         for subfolder in os.listdir(base_path):
#             full_subfolder_path = os.path.join(base_path, subfolder)
#             if os.path.isdir(full_subfolder_path):
#                 # 定义要删除的路径
#                 screenshot_draw_path = os.path.join(full_subfolder_path, 'backup')
#                 # task_infos_path = os.path.join(full_subfolder_path, 'screenshot')
#                 # crop_path = os.path.join(full_subfolder_path, 'actions.json')
#
#                 # 删除 screenshot_draw 文件夹
#                 if os.path.isdir(screenshot_draw_path):
#                     shutil.rmtree(screenshot_draw_path)
#                     print(f"已删除文件夹: {screenshot_draw_path}")

                # # 删除 crop 文件夹
                # if os.path.isdir(crop_path):
                #     shutil.rmtree(crop_path)
                #     print(f"已删除文件夹: {crop_path}")
                #
                # # 删除 task_infos.json 文件
                # if os.path.isfile(task_infos_path):
                #     os.remove(task_infos_path)
                #     print(f"已删除文件: {task_infos_path}")
#     else:
#         print(f"路径不存在: {base_path}")
#
# print("操作完成")

# import os
#
# def delete_mp4_files_in_subfolders(base_dir):
#     # 遍历base_dir目录下的所有子文件夹
#     for root, dirs, files in os.walk(base_dir):
#         # 查找每个子文件夹中的MP4文件
#         for file in files:
#             if file.lower().endswith(".mp4"):  # 匹配MP4文件，忽略大小写
#                 file_path = os.path.join(root, file)
#                 try:
#                     os.remove(file_path)  # 删除文件
#                     print(f"Deleted: {file_path}")
#                 except Exception as e:
#                     print(f"Error deleting {file_path}: {e}")
#
# # 调用函数，传入目标文件夹路径
# base_directory = "data/general_new/qq"  # 这里填写你的文件夹路径
# delete_mp4_files_in_subfolders(base_directory)
import os
import shutil

# 定义要删除的文件名
files_to_delete = ['operation_ui_graph', 'record.mp4']


# 遍历data/general_EN目录
def delete_files_in_directory(root_dir):
    for app_dir in os.listdir(root_dir):
        app_path = os.path.join(root_dir, app_dir)

        # 确保这是一个目录
        if os.path.isdir(app_path):
            for scene_dir in os.listdir(app_path):
                scene_path = os.path.join(app_path, scene_dir)

                # 确保这是一个目录
                if os.path.isdir(scene_path):
                    # 遍历每个场景中的文件
                    for file_name in os.listdir(scene_path):
                        file_path = os.path.join(scene_path, file_name)

                        # 如果文件名匹配要删除的文件，删除它
                        if file_name in files_to_delete:
                            try:
                                if os.path.isdir(file_path):
                                    shutil.rmtree(file_path)  # 如果是目录，使用rmtree删除
                                else:
                                    os.remove(file_path)  # 如果是文件，使用remove删除
                                print(f"Deleted: {file_path}")
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}")


# 设置根目录路径
root_directory = 'general_EN'

# 调用函数进行删除操作
delete_files_in_directory(root_directory)
