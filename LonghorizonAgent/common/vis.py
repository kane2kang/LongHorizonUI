import pdb

import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import os
import platform

from . import utils


def visualize_ocr_results(image_bgr, ocr_results):
    """
    将 OCR 结果（边界框和文本）绘制在与原始图像相同大小的白色图像上，
    并将结果与原始图像拼接后保存。

    Args:
        image_bgr (numpy array): image array
        ocr_results (list): OCR 结果列表，每个元素包含边界框坐标和文本。
    """
    height, width, _ = image_bgr.shape

    white_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    for result in ocr_results:
        box = result[0]
        text = result[1]
        score = result[2]
        try:
            # 绘制边界框
            # cv2.polylines(white_image, [np.array(box).astype(np.int32)], True, (0, 255, 0), 2)

            cv2.rectangle(white_image, np.array(box[0]).astype(np.int32).tolist(),
                          np.array(box[2]).astype(np.int32).tolist(), (0, 255, 0), 2)

            # 计算边界框的宽度和高度
            box_width = int(np.linalg.norm(np.array(box[0]) - np.array(box[1])))
            box_height = int(np.linalg.norm(np.array(box[1]) - np.array(box[2])))

            font_size = max(1, int(min(box_width, box_height) * 0.5))  # 可以根据需要调整比例因子
            font = utils.get_font(font_size)

            # 将 OpenCV 图像转换为 Pillow 图像
            img_pil = Image.fromarray(white_image)
            draw = ImageDraw.Draw(img_pil)

            # 计算文本绘制位置，使其在边界框内居中
            text_x = int(box[0][0] + box_height / 4)
            text_y = int(box[0][1] + box_height / 4)

            # 绘制文本
            draw.text((text_x, text_y), text, font=font, fill=(0, 0, 255, 255))

            # 将 Pillow 图像转换回 OpenCV 图像
            white_image = np.array(img_pil)
        except Exception as e:
            continue

    # 将原始图像和绘制结果的图像水平拼接
    if height > width:
        image_draw = np.concatenate((image_bgr, white_image), axis=1)
    else:
        image_draw = np.concatenate((image_bgr, white_image), axis=0)
    return image_draw
