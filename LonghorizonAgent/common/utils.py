import pdb

import torch
import cv2
import base64
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import traceback
import io
from cryptography.fernet import Fernet
from io import StringIO
from dotenv import dotenv_values
import json
import hashlib

def generate_unique_md5(input_str):
    """
    根据字符串生成md5
    :param input_str: 输入字符串
    :return:
    """
    md5 = hashlib.md5()
    md5.update(input_str.encode('utf-8'))
    unique_md5 = md5.hexdigest()
    return unique_md5

def get_optimize_device_and_dtype():
    """
    获取最优的pytorch运行的device和dtype
    :return:
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("cpu"), torch.float32
    else:
        return torch.device("cpu"), torch.float32


def calculate_size(box):
    """
    计算box的面积
    :param box: 输入box
    :return:
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_iou(box1, box2):
    """
    计算两个box之间的iou
    :param box1:
    :param box2:
    :return:
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    iou = interArea / (unionArea + 1e-08)

    return iou


def calculate_iou_v2(box1, box2):
    """
    计算两个box的iou, v2版本返回的是分别在两个box上的iou
    :param box1:
    :param box2:
    :return:
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou1 = interArea / (box1Area + 1e-08)
    iou2 = interArea / (box2Area + 1e-08)
    return iou1, iou2


def remove_boxes(boxes_filt, size, iou_threshold=0.4):
    """
    去掉面积比较小或是重叠度比较高的box
    :param boxes_filt:
    :param size: 屏幕尺寸
    :param iou_threshold:
    :return:
    """
    boxes_to_remove = set()

    for i in range(len(boxes_filt)):
        for j in range(len(boxes_filt)):
            if i == j:
                continue
            if i in boxes_to_remove or j in boxes_to_remove:
                continue
            iou1, iou2 = calculate_iou_v2(boxes_filt[i], boxes_filt[j])
            if max(iou1, iou2) >= iou_threshold:
                if iou1 > iou2:
                    boxes_to_remove.add(j)
                else:
                    boxes_to_remove.add(i)

    boxes_filt = [box for idx, box in enumerate(boxes_filt) if idx not in boxes_to_remove]
    return boxes_filt


def merge_icon_bboxs_v2(icon_bboxs1, icon_bboxs2, iou_threshold=0.5):
    """
    把所有的icon box都合并起来
    :param icon_bboxs1:
    :param icon_bboxs2:
    :param iou_threshold:
    :return:
    """
    merged_boxes = []
    used_indices2 = set()

    for box1 in icon_bboxs1:
        for i, box2 in enumerate(icon_bboxs2):
            if i in used_indices2:
                continue
            iou1, iou2 = calculate_iou_v2(box1, box2)
            if max(iou1, iou2) > iou_threshold:
                used_indices2.add(i)
                break
        merged_boxes.append(box1)

    # Add any remaining boxes from icon_bboxs2 that were not merged
    for i, box2 in enumerate(icon_bboxs2):
        if i not in used_indices2:
            merged_boxes.append(box2)

    return merged_boxes


def encode_image_to_base64(image, image_format='png'):
    """
    Encodes a OpenCV image to a Base64 string.

    Args:
        image (numpy.ndarray): The OpenCV image.
        image_format (str): The image format to use for encoding (e.g., 'png', 'jpeg').

    Returns:
        str: The Base64 encoded string of the image, or None if encoding fails.
    """
    try:
        # Convert the image to a byte array
        _, buffer = cv2.imencode(f'.{image_format}', image)

        # Encode the byte array to Base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')  # Decode to a string

        return image_base64
    except Exception as e:
        traceback.print_exc()
        return None


def decode_base64_to_image(base64_string):
    """
    Decodes a Base64 string to an OpenCV image.

    Args:
        base64_string (str): The Base64 encoded string of the image.

    Returns:
        numpy.ndarray: The OpenCV image, or None if decoding fails.
    """
    try:
        # Decode the Base64 string to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert the bytes to a numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode the numpy array to an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # or cv2.IMREAD_GRAYSCALE

        return image
    except Exception as e:
        traceback.print_exc()
        return None


def base64_to_pil_image(base64_string: str) -> Image.Image | None:
    """Converts a base64 string to a PIL Image object."""
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert('RGB')  # Use RGBA for drawing
    except Exception as e:
        traceback.print_exc()
        return None


def pil_image_to_base64(image_pil: Image.Image) -> str | None:
    """Converts a PIL Image object to a base64 string."""
    try:
        buffered = io.BytesIO()
        # Ensure saving as PNG to preserve transparency if used
        image_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        traceback.print_exc()
        return None


def encode_image(image_path):
    """
    图片编码成base64
    :param image_path:
    :return:
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def decode_image(base64_string: str):
    """Decodes a base64 string into a BytesIO image stream."""
    try:
        image_data = base64.b64decode(base64_string)
        return io.BytesIO(image_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def image_numpy_to_base64(image):
    """
    将OpenCV读取的BGR图像转换为base64字符串

    Args:
        image: OpenCV读取的BGR格式图像

    Returns:
        str: base64编码的图像字符串
    """
    _, buffer = cv2.imencode('.jpg', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return base64_str


def base64_to_image_numpy(base64_string):
    """
    将base64编码的图像字符串转换为OpenCV BGR numpy数组。

    Args:
      base64_string:  base64编码的图像字符串。

    Returns:
      一个OpenCV BGR numpy数组，如果转换成功；否则返回None。
    """
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def add_grid_with_numbers(image, **kwargs):
    """
    Adds a grid with row and column numbers on all four sides to an image.

    Args:
        image (np.ndarray): Input image as a NumPy array (assumed RGB).
        **kwargs:
            grid_num (int): Approximate number of grid lines along the shorter image dimension. Defaults to 12.
            font_path (str): Optional path to a .ttf font file.
            font_scale (float): Factor to scale font size relative to grid size. Defaults to 0.75.
            min_font_size (int): Minimum font size in pixels. Defaults to 10.
            line_color (str): Color of the grid lines. Defaults to "black".
            text_color (str): Color of the number text. Defaults to "black".
            bg_color (str): Color of the border background. Defaults to "white".

    Returns:
        np.ndarray: Image with grid and numbers as a NumPy array (RGB).
    """
    # --- Get Parameters ---
    grid_num = kwargs.get("grid_num", 11)
    font_path_user = kwargs.get("font_path", None)
    font_scale = kwargs.get("font_scale", 0.75)
    min_font_size = kwargs.get("min_font_size", 10)
    line_color = kwargs.get("line_color", "black")
    text_color = kwargs.get("text_color", "black")
    bg_color = kwargs.get("bg_color", "white")

    # --- Image Setup ---
    img = Image.fromarray(image)
    width, height = img.size
    if width == 0 or height == 0:
        print("Error: Input image has zero width or height.")
        return image  # Return original if invalid

    # --- Grid Calculation ---
    grid_num = max(1, grid_num)
    # Use grid_size as the border size
    border_size = max(1, min(width, height) // grid_num)  # Ensure border_size is at least 1

    # --- Create New Image with Border on ALL sides ---
    # Add border_size to both left/right and top/bottom
    new_width = width + 2 * border_size
    new_height = height + 2 * border_size
    max_size = max(height, width) + 2 * border_size
    new_img = Image.new('RGB', (new_width, new_height), bg_color)
    # Paste original image offset by the top-left border
    new_img.paste(img, (border_size, border_size))

    # --- Drawing Context ---
    draw = ImageDraw.Draw(new_img)

    # --- Font Loading ---
    font_size = max(min_font_size, int(border_size * font_scale))
    font = get_font(font_size)

    # --- Helper Function for Centering Text (Refined) ---
    def get_centered_text_pos(box_x, box_y, box_w, box_h, text, current_font):
        if current_font is None: return (box_x, box_y)

        try:
            if hasattr(current_font, "getbbox"):
                if hasattr(current_font, "getlength"):
                    text_width = current_font.getlength(text)
                else:
                    text_width, _ = current_font.getsize(text)  # Fallback getsize

                if hasattr(current_font, "getmetrics"):
                    ascent, descent = current_font.getmetrics()
                    text_height = ascent + descent
                else:
                    # Use font_size as approximation if metrics not available
                    text_height = font_size * 1.1  # Approx height
            else:
                text_width, text_height = current_font.getsize(text)  # Older PIL/Pillow or default

        except AttributeError:
            print("Warning: Font object missing expected methods for size calculation.")
            text_width, text_height = 10, 10  # Default fallback size

        text_x = box_x + (box_w - text_width) / 2
        text_y = box_y + (box_h - text_height) / 2
        return int(text_x), int(text_y)

    # 颜色调色板，用于单元格中心的交叉线和序号背景
    color_palette = [
        "red", "green", "blue", "yellow", "purple", "orange",
        "cyan", "magenta", "lime", "pink", "teal", "brown",
        "navy", "coral", "gold", "grey", "indigo"
    ]

    # --- Draw Grid Numbers with Colored Background ---
    if font:
        # 绘制顶部列号（带颜色背景）
        for i in range(0, width - border_size // 2, border_size):
            col_index = i // border_size
            grid_number = str(col_index + 1)
            cell_x = border_size + i
            cell_y = 0
            color = color_palette[col_index % len(color_palette)]

            # 绘制背景色块
            draw.rectangle([cell_x, cell_y, cell_x + border_size, cell_y + border_size], fill=color)

            # 绘制数字
            text_x, text_y = get_centered_text_pos(cell_x, cell_y, border_size, border_size, grid_number, font)
            draw.text((text_x, text_y), grid_number, fill=text_color, font=font)

        # 绘制左侧行号（带颜色背景）
        for j in range(0, height - border_size // 2, border_size):
            row_index = j // border_size
            grid_number = str(row_index + 1)
            cell_x = 0
            cell_y = border_size + j
            color = color_palette[row_index % len(color_palette)]

            # 绘制背景色块
            draw.rectangle([cell_x, cell_y, cell_x + border_size, cell_y + border_size], fill=color)

            # 绘制数字
            text_x, text_y = get_centered_text_pos(cell_x, cell_y, border_size, border_size, grid_number, font)
            draw.text((text_x, text_y), grid_number, fill=text_color, font=font)

        # 绘制底部列号（带颜色背景）
        for i in range(0, width - border_size // 2, border_size):
            col_index = i // border_size
            grid_number = str(col_index + 1)
            cell_x = border_size + i
            cell_y = new_height - border_size
            color = color_palette[col_index % len(color_palette)]

            # 绘制背景色块
            draw.rectangle([cell_x, cell_y, cell_x + border_size, cell_y + border_size], fill=color)

            # 绘制数字
            text_x, text_y = get_centered_text_pos(cell_x, cell_y, border_size, border_size, grid_number, font)
            draw.text((text_x, text_y), grid_number, fill=text_color, font=font)

        # 绘制右侧行号（带颜色背景）
        for j in range(0, height - border_size // 2, border_size):
            row_index = j // border_size
            grid_number = str(row_index + 1)
            cell_x = new_width - border_size
            cell_y = border_size + j
            color = color_palette[row_index % len(color_palette)]

            # 绘制背景色块
            draw.rectangle([cell_x, cell_y, cell_x + border_size, cell_y + border_size], fill=color)

            # 绘制数字
            text_x, text_y = get_centered_text_pos(cell_x, cell_y, border_size, border_size, grid_number, font)
            draw.text((text_x, text_y), grid_number, fill=text_color, font=font)

    # --- 绘制黑色网格线 ---
    # 垂直黑线
    for i, j in enumerate(range(border_size * 2, border_size + width + 1, border_size)):
        line_x = min(j, border_size + width)
        draw.line([(line_x, 0), (line_x, new_height)], fill=color_palette[i % len(color_palette)], width=4)

    # for i, j in enumerate(range(border_size + border_size // 2, border_size + width + 1, border_size)):
    #     line_x = min(j, border_size + width)
    #     draw.line([(line_x, border_size), (line_x, new_height - border_size)],
    #               fill=color_palette[i % len(color_palette)], width=4)

    # 水平黑线
    for i, j in enumerate(range(border_size * 2, border_size + height + 1, border_size)):
        line_y = min(j, border_size + height)
        draw.line([(0, line_y), (new_width, line_y)], fill=color_palette[i % len(color_palette)], width=4)

    # for i, j in enumerate(range(border_size + border_size // 2, border_size + height + 1, border_size)):
    #     line_y = min(j, border_size + height)
    #     draw.line([(border_size, line_y), (new_width - border_size, line_y)],
    #               fill=color_palette[i % len(color_palette)], width=4)

    open_cv_image = np.array(new_img)
    return open_cv_image


def add_grid_with_numbers_dual_highlight(image, **kwargs):
    """
    Creates two versions of an input image with grid numbers - one highlighting rows and
    one highlighting columns - and concatenates them into a single output image.
    Semi-transparent highlighting extends to cover the number labels as well.

    Args:
        image (np.ndarray): Input image as a NumPy array (assumed RGB).
        **kwargs:
            grid_num (int): Approximate number of grid lines along the shorter image dimension. Defaults to 11.
            font_path (str): Optional path to a .ttf font file.
            font_scale (float): Factor to scale font size relative to grid size. Defaults to 0.75.
            min_font_size (int): Minimum font size in pixels. Defaults to 10.
            line_color (str): Color of the grid lines. Defaults to "black".
            text_color (str): Color of the number text. Defaults to "black".
            bg_color (str): Color of the border background. Defaults to "white".
            highlight_alpha (float): Alpha transparency for row/column highlighting. Defaults to 0.3.
            gap_size (int): Gap between the two images in pixels. Defaults to 20.

    Returns:
        np.ndarray: Concatenated image with grid and numbers as a NumPy array (RGB).
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageColor
    import os

    # --- Get Parameters ---
    grid_num = kwargs.get("grid_num", 11)
    font_path_user = kwargs.get("font_path", None)
    font_scale = kwargs.get("font_scale", 0.75)
    min_font_size = kwargs.get("min_font_size", 10)
    line_color = kwargs.get("line_color", "black")
    text_color = kwargs.get("text_color", "black")
    bg_color = kwargs.get("bg_color", "white")
    highlight_alpha = kwargs.get("highlight_alpha", 0.3)
    gap_size = kwargs.get("gap_size", 20)  # Gap between images

    # Color palette for highlighting rows and columns
    color_palette = [
        "red", "green", "blue", "purple", "orange",
        "cyan", "magenta", "lime", "teal", "brown",
        "navy", "coral", "gold", "grey", "darkgreen", "indigo"
    ]

    def process_image(highlight_rows=False, highlight_cols=False):
        """Helper function to process the image with appropriate highlighting"""
        # --- Image Setup ---
        img = Image.fromarray(image)
        width, height = img.size
        if width == 0 or height == 0:
            print("Error: Input image has zero width or height.")
            return image  # Return original if invalid

        # --- Grid Calculation ---
        grid_num_actual = max(1, grid_num)
        # Use grid_size as the border size
        border_size = max(1, min(width, height) // grid_num_actual)  # Ensure border_size is at least 1

        # --- Create New Image with Border on ALL sides ---
        new_width = width + 2 * border_size
        new_height = height + 2 * border_size
        new_img = Image.new('RGBA', (new_width, new_height), bg_color)

        # Paste original image offset by the top-left border
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        new_img.paste(img, (border_size, border_size), img if img.mode == 'RGBA' else None)

        # --- Drawing Context ---
        draw = ImageDraw.Draw(new_img)

        # --- Font Loading ---
        font_size = max(min_font_size, int(border_size * font_scale))
        font = get_font(font_size)

        # --- Helper Function for Centering Text ---
        def get_centered_text_pos(box_x, box_y, box_w, box_h, text, current_font):
            if current_font is None: return (box_x, box_y)

            try:
                if hasattr(current_font, "getbbox"):
                    if hasattr(current_font, "getlength"):
                        text_width = current_font.getlength(text)
                    else:
                        text_width, _ = current_font.getsize(text)  # Fallback getsize

                    if hasattr(current_font, "getmetrics"):
                        ascent, descent = current_font.getmetrics()
                        text_height = ascent + descent
                    else:
                        # Use font_size as approximation if metrics not available
                        text_height = font_size * 1.1  # Approx height
                else:
                    text_width, text_height = current_font.getsize(text)  # Older PIL/Pillow or default

            except AttributeError:
                print("Warning: Font object missing expected methods for size calculation.")
                text_width, text_height = 10, 10  # Default fallback size

            text_x = box_x + (box_w - text_width) / 2
            text_y = box_y + (box_h - text_height) / 2
            return int(text_x), int(text_y)

        # --- Draw Row/Column Highlighting ---
        # Create a transparent overlay for highlighting
        overlay = Image.new('RGBA', new_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Apply row highlighting (including numbers on left and right)
        if highlight_rows:
            for j in range(0, height - border_size // 2, border_size):
                row_index = j // border_size
                row_color = ImageColor.getrgb(color_palette[row_index % len(color_palette)])
                # Convert RGB to RGBA with transparency
                row_color_alpha = (row_color[0], row_color[1], row_color[2], int(255 * highlight_alpha))

                # Draw semi-transparent row highlight (including left and right number areas)
                overlay_draw.rectangle(
                    [0, border_size + j,
                     new_width, border_size + j + border_size],
                    fill=row_color_alpha
                )

        # Apply column highlighting (including numbers on top and bottom)
        if highlight_cols:
            for i in range(0, width - border_size // 2, border_size):
                col_index = i // border_size
                col_color = ImageColor.getrgb(color_palette[col_index % len(color_palette)])
                # Convert RGB to RGBA with transparency
                col_color_alpha = (col_color[0], col_color[1], col_color[2], int(255 * highlight_alpha))

                # Draw semi-transparent column highlight (including top and bottom number areas)
                overlay_draw.rectangle(
                    [border_size + i, 0,
                     border_size + i + border_size, new_height],
                    fill=col_color_alpha
                )

        # Composite the overlay with the main image
        new_img = Image.alpha_composite(new_img, overlay)
        draw = ImageDraw.Draw(new_img)

        # --- Draw Grid Numbers ---
        if font:
            # Draw top column numbers
            for i in range(0, width - border_size // 2, border_size):
                col_index = i // border_size
                grid_number = str(col_index + 1)
                cell_x = border_size + i
                cell_y = 0

                # Draw number (no background, as it's now covered by the semi-transparent highlight)
                text_x, text_y = get_centered_text_pos(cell_x, cell_y, border_size, border_size, grid_number, font)
                draw.text((text_x, text_y), grid_number, fill=text_color, font=font)

            # Draw left row numbers
            for j in range(0, height - border_size // 2, border_size):
                row_index = j // border_size
                grid_number = str(row_index + 1)
                cell_x = 0
                cell_y = border_size + j

                # Draw number (no background, as it's now covered by the semi-transparent highlight)
                text_x, text_y = get_centered_text_pos(cell_x, cell_y, border_size, border_size, grid_number, font)
                draw.text((text_x, text_y), grid_number, fill=text_color, font=font)

            # Draw bottom column numbers
            for i in range(0, width - border_size // 2, border_size):
                col_index = i // border_size
                grid_number = str(col_index + 1)
                cell_x = border_size + i
                cell_y = new_height - border_size

                # Draw number (no background, as it's now covered by the semi-transparent highlight)
                text_x, text_y = get_centered_text_pos(cell_x, cell_y, border_size, border_size, grid_number, font)
                draw.text((text_x, text_y), grid_number, fill=text_color, font=font)

            # Draw right row numbers
            for j in range(0, height - border_size // 2, border_size):
                row_index = j // border_size
                grid_number = str(row_index + 1)
                cell_x = new_width - border_size
                cell_y = border_size + j

                # Draw number (no background, as it's now covered by the semi-transparent highlight)
                text_x, text_y = get_centered_text_pos(cell_x, cell_y, border_size, border_size, grid_number, font)
                draw.text((text_x, text_y), grid_number, fill=text_color, font=font)

        # --- Draw Black Grid Lines ---
        # Vertical lines
        for i in range(border_size, border_size + width + 1, border_size):
            line_x = min(i, border_size + width)
            draw.line([(line_x, 0), (line_x, new_height)], fill=line_color, width=2)

        # Horizontal lines
        for j in range(border_size, border_size + height + 1, border_size):
            line_y = min(j, border_size + height)
            draw.line([(0, line_y), (new_width, line_y)], fill=line_color, width=2)

        # Convert back to RGB for compatibility
        new_img = new_img.convert('RGB')
        return np.array(new_img)

    # Process the image with two different highlighting styles
    img_rows = process_image(highlight_rows=True, highlight_cols=False)
    img_cols = process_image(highlight_rows=False, highlight_cols=True)

    # Determine concatenation direction based on image aspect ratio
    h, w = image.shape[:2]
    aspect_ratio = h / w

    # Create a background image for concatenation with a gap
    if aspect_ratio > 1:  # h > w, concatenate horizontally (left to right)
        total_width = img_rows.shape[1] + img_cols.shape[1] + gap_size
        total_height = max(img_rows.shape[0], img_cols.shape[0])

        # Create background image
        bg_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

        # Paste the first image (row highlighting)
        bg_img[:img_rows.shape[0], :img_rows.shape[1]] = img_rows

        # Paste the second image (column highlighting)
        bg_img[:img_cols.shape[0], img_rows.shape[1] + gap_size:] = img_cols
    else:  # h ≤ w, concatenate vertically (top to bottom)
        total_width = max(img_rows.shape[1], img_cols.shape[1])
        total_height = img_rows.shape[0] + img_cols.shape[0] + gap_size

        # Create background image
        bg_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

        # Paste the first image (row highlighting)
        bg_img[:img_rows.shape[0], :img_rows.shape[1]] = img_rows

        # Paste the second image (column highlighting)
        bg_img[img_rows.shape[0] + gap_size:, :img_cols.shape[1]] = img_cols

    return bg_img


def get_font(font_size):
    font = None
    # Try to load nicer fonts
    try:
        # Try different font options in order of preference
        font_options = ['Songti', 'SongTi', 'simsun', 'Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
        font_loaded = False

        for font_name in font_options:
            try:
                import platform
                if platform.system().lower() == 'windows':
                    # Need to specify the abs font path on Windows
                    font_path = os.path.join(os.getenv('WINDIR', 'C:\\Windows'), "Fonts", font_name + '.ttf')
                    if not os.path.exists(font_path) and os.getenv("LOCALAPPDATA"):
                        font_path = os.path.join(os.getenv('LOCALAPPDATA'), "Microsoft", "Windows", "Fonts",
                                                 font_name + '.ttf')
                else:
                    font_path = font_name
                font = ImageFont.truetype(font_path, font_size)
                font_loaded = True
                break
            except OSError:
                continue

        if not font_loaded:
            print("No font found!")
            raise OSError('No preferred fonts found')

    except OSError:
        font = ImageFont.load_default()
        print("No preferred fonts found")
    return font

def get_font_chinese(size: int) -> ImageFont.FreeTypeFont:
    """获取支持多语言字体的方法，带有多重回退机制"""
    font_paths = [
        # 常见系统字体路径（按优先级排序）
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',  # macOS
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        'C:/Windows/Fonts/msyh.ttc',  # Windows 雅黑
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Linux Noto
        # 添加更多字体路径...
    ]

    # 尝试所有可用字体路径
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except IOError:
                continue

    # 最终回退到Pillow默认字体
    return ImageFont.load_default()


def load_env_from_encrypted_file(encrypted_file_path):
    """
    Decrypts the contents of an encrypted .env file using the Fernet key and loads
    the environment variables into os.environ.

    Args:
        encrypted_file_path (str): The path to the encrypted .env file.
        key (bytes): The Fernet key used for decryption.
    """
    try:
        with open(encrypted_file_path, "rb") as encrypted_file:
            encrypted_data = encrypted_file.read()

        ENCRYPT_KEY = os.getenv("QAGENT-OMNI-ENCRYPT-KEY", "gDJVDmhGris4Lal7iPH8JdE2K9fqwHovnpT3GZ8qjMw=")
        f = Fernet(ENCRYPT_KEY)
        decrypted_bytes = f.decrypt(encrypted_data)
        decrypted_string = decrypted_bytes.decode('utf-8')

        # Load environment variables directly from the string using dotenv_values
        env_vars = dotenv_values(stream=StringIO(decrypted_string))

        # Set the environment variables in os.environ
        for key, value in env_vars.items():
            os.environ[key] = value

        print("Environment variables loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Encrypted .env file not found: {encrypted_file_path}")
    except Exception as e:
        print(f"Error loading environment variables: {e}")


def load_json_from_encrypted_file(encrypted_file_path):
    """
    Decrypts the contents of an encrypted JSON file using the Fernet key and loads
    the JSON data into a Python dictionary.

    Args:
        encrypted_file_path (str): The path to the encrypted JSON file.
        key (bytes): The Fernet key used for decryption.

    Returns:
        dict: The decrypted JSON data as a Python dictionary, or None if an error occurs.
    """
    try:
        with open(encrypted_file_path, "rb") as encrypted_file:
            encrypted_data = encrypted_file.read()

        ENCRYPT_KEY = os.getenv("QAGENT-OMNI-ENCRYPT-KEY", "gDJVDmhGris4Lal7iPH8JdE2K9fqwHovnpT3GZ8qjMw=")
        f = Fernet(ENCRYPT_KEY)
        decrypted_bytes = f.decrypt(encrypted_data)
        decrypted_string = decrypted_bytes.decode('utf-8')

        # Load the JSON data from the decrypted string
        json_data = json.loads(decrypted_string)

        print("JSON data loaded successfully.")
        return json_data

    except FileNotFoundError:
        print(f"Error: Encrypted JSON file not found: {encrypted_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON data in decrypted file: {encrypted_file_path}")
        return None
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None
