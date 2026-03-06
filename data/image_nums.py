import os

# 根路径
base_path = 'general'

# 支持的图像扩展名
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

# 总图像数
total_images = 0

# 存储每个场景的图像数量
scene_image_counts = {}

# 遍历所有子文件夹（即每个场景）
for scene_name in os.listdir(base_path):
    scene_path = os.path.join(base_path, scene_name)
    if os.path.isdir(scene_path):
        screenshot_path = os.path.join(scene_path, 'screenshot')
        image_count = 0

        if os.path.isdir(screenshot_path):
            for file_name in os.listdir(screenshot_path):
                # 检查是否为图像，且文件名不包含 "highlight"
                if (os.path.splitext(file_name)[1].lower() in image_extensions
                        and 'highlight' not in file_name.lower()):
                    image_count += 1

        scene_image_counts[scene_name] = image_count
        total_images += image_count

# 输出每个场景的图像数量
for scene, count in scene_image_counts.items():
    print(f"{scene}: {count} 张图像（不包含 'highlight'）")

# 输出总图像数量
print(f"\n总图像数量（不包含 'highlight'）: {total_images}")
