import copy
import json
import logging
import os
import pdb
import random
import re
import time
import uuid
from dataclasses import dataclass, field, Field
from tkinter import Image
from typing import Literal, Any, Tuple
from typing import TYPE_CHECKING, Optional, TypedDict, List, Dict
from PIL import Image, ImageDraw, ImageFont
import math
import cv2
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path
from ..common import utils
from ..perception.screen_perception import PerceptionInfo


@dataclass
class SystemState:
    screenshot_dim: tuple | None = None
    screenshot_path: str | None = None
    screenshot_base64: str | None = None
    highlight_screenshot_path: str | None = None
    highlight_screenshot_base64: str | None = None
    highlight_action_path: str | None = None
    highlight_action_base64: str | None = None
    perception_infos: PerceptionInfo | None = field(default_factory=PerceptionInfo)
    perception_description: str | None = None


@dataclass
class SystemContextConfig:
    platform: str | None = None

    highlight_elements: bool = True
    highlight_type: str | None = "normal"
    highlight_grid_num: int = 11

    screenshot_save_dir: str = "./tmp/screenshots"
    icon_template_dir: str | None = None

    # perception config
    use_perception: bool = True
    perception_type: Literal['local', 'api'] = "local"
    perception_description_type: Literal['md', 'normal'] = "md"
    perception_server_url: str | None = None
    use_ocr: bool = True
    use_ocr_rec: bool = False
    use_icon_detect: bool = True
    use_template_match: bool = True
    use_icon_caption: bool = False
    detect_split_x: int | None = None
    detect_split_y: int | None = None
    detect_type: Literal['combined', 'split', 'single'] = "combined"
    detect_grid_num: int = 16
    detect_imgsz: int = 640

    # usually for browser web page
    ocr_split_text: bool = False
    # OCR result patch
    # ocr_results_path: str = field(
    #     default=None,
    #     metadata={"help": "Path for saving OCR results in JSON format"}
    # )


logger = logging.getLogger(__name__)


class SystemContext:
    def __init__(self, config: SystemContextConfig = SystemContextConfig()):
        self.config = config
        self.perception_model = None
        self.perception_client = None
        self.cached_state = SystemState()
        self.init_perception()
        self.init_context()

    def init_context(self):
        """
        初始化context
        :return:
        """
        pass

    def init_perception(self):
        """
        初始化感知模块模型
        :return:
        """
        if self.config.use_perception:
            if self.config.perception_type == "local":
                from ..perception.screen_perception import ScreenPerception
                self.perception_model = ScreenPerception(use_icon_caption=self.config.use_icon_caption,
                                                         icon_template_dir=self.config.icon_template_dir,
                                                         use_rec=self.config.use_ocr_rec
                                                         )
            else:
                pass

    def take_screenshot(self, image_path: Optional[str]=None) -> str:
        """
        获取屏幕截图, 继承类必须要实现
        :return:
        """
        raise NotImplemented

    def _generate_distinct_colors(self, n):
        """
        Generates n visually distinct colors.

        Args:
            n: Number of colors to generate

        Returns:
            List of BGR color tuples
        """
        colors = []

        colors_shuffle = list(range(n))
        random.shuffle(colors_shuffle)
        for i in colors_shuffle:
            # Use HSV color space to generate distinct colors
            # Vary hue from 0 to 179 (OpenCV's H range)
            hue = int(179 * i / n)
            # Use high saturation and value for vibrant colors
            saturation = random.randint(128, 255)
            value = random.randint(128, 255)

            # Convert HSV to BGR
            hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]

            # Convert to regular tuple
            colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))

        return colors

    def highlight(self, image, screen_state: PerceptionInfo, **kwargs):
        """
        Visualizes perception information on the image using PIL.ImageDraw.

        Args:
            image: The original PIL Image to draw on.
            screen_state: Screen state containing perception information (assumed to have .perception_info).
            **kwargs: Additional arguments (unused in this core logic).

        Returns:
            PIL Image with drawn perception information, or None if input is invalid.
        """
        if image is None or screen_state is None:
            return image

        perception_info = screen_state.perception_info
        if not perception_info:
            print("Warning: No perception_info found in screen_state.")
            return image  # Return original if no info to draw

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure the image is in RGBA format
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Create a transparent overlay for semi-transparent rectangles
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        image_width, image_height = image.size

        # Create another copy for outlines and text
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        # Generate distinct colors for each box
        num_boxes = len(perception_info)
        colors = self._generate_distinct_colors(num_boxes)

        # Define drawing parameters
        alpha_percent = 0.3  # Transparency factor
        outline_thickness = 2
        label_margin = 1  # Margin for label placement

        # Helper function to get dominant color in a region
        def get_dominant_color(img, box, sample_size=10):
            """Extract the dominant color from a region of the image"""
            x1, y1, x2, y2 = box
            # Sample area where text will be placed (typically top-left for an index)
            # Ensure the sample box is within the image boundaries
            sample_x1 = max(0, x1)
            sample_y1 = max(0, y1)
            sample_x2 = min(img.width, x1 + sample_size)
            sample_y2 = min(img.height, y1 + sample_size)

            # If the calculated sample area has zero width or height,
            # try sampling from the center of the original box, or return a default.
            if sample_x2 <= sample_x1 or sample_y2 <= sample_y1:
                # Fallback: try a small region from the center of the original box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                half_sample = sample_size // 2
                sample_x1 = max(0, center_x - half_sample)
                sample_y1 = max(0, center_y - half_sample)
                sample_x2 = min(img.width, center_x + half_sample)
                sample_y2 = min(img.height, center_y + half_sample)
                if sample_x2 <= sample_x1 or sample_y2 <= sample_y1:
                    return (128, 128, 128)  # Default if still no valid region

            text_region = img.crop((sample_x1, sample_y1, sample_x2, sample_y2))

            if text_region.mode != 'RGB':
                text_region = text_region.convert('RGB')

            if text_region.size[0] == 0 or text_region.size[1] == 0:
                return (128, 128, 128)  # Default if cropped region is empty

            color_data = np.array(text_region)
            avg_color = np.mean(color_data, axis=(0, 1)).astype(int)
            return tuple(np.clip(avg_color, 0, 255))  # Ensure colors are within 0-255 range

        # Helper functions for WCAG contrast calculation
        def srgb_to_linear(c_srgb):
            """Convert sRGB value (0-1) to linear RGB value."""
            if c_srgb <= 0.04045:
                return c_srgb / 12.92
            else:
                return ((c_srgb + 0.055) / 1.055) ** 2.4

        def get_relative_luminance(rgb_tuple):
            """Calculate relative luminance for an RGB tuple (0-255)."""
            r, g, b = [srgb_to_linear(c / 255.0) for c in rgb_tuple]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def get_contrast_ratio(lum1, lum2):
            """Calculate contrast ratio between two relative luminances."""
            if lum1 > lum2:
                return (lum1 + 0.05) / (lum2 + 0.05)
            else:
                return (lum2 + 0.05) / (lum1 + 0.05)

        def get_contrasting_color(background_rgb_tuple):
            """
            Return a color with high contrast to the input background_rgb_tuple.
            Tests black, white, and a few other candidates for best WCAG contrast.
            """
            bg_lum = get_relative_luminance(background_rgb_tuple)

            candidate_colors = {
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "bright_yellow": (255, 255, 0),  # Good for dark backgrounds
                "dark_blue": (0, 0, 139),  # Good for light/yellowish backgrounds
                "bright_cyan": (0, 255, 255),  # Good for dark/reddish backgrounds
                "dark_red": (139, 0, 0),  # Good for light/greenish backgrounds
                # Add more candidates if desired
                # "light_gray": (200, 200, 200),
                # "dark_gray": (70, 70, 70),
            }

            best_color_tuple = (0, 0, 0)  # Default to black
            max_contrast = 0

            for name, color_tuple in candidate_colors.items():
                fg_lum = get_relative_luminance(color_tuple)
                contrast = get_contrast_ratio(bg_lum, fg_lum)
                # print(f"Contrast with {name} {color_tuple}: {contrast:.2f}") # For debugging
                if contrast > max_contrast:
                    max_contrast = contrast
                    best_color_tuple = color_tuple
            return best_color_tuple

        # Draw each box with its label
        for i, info in enumerate(perception_info):
            if "box" not in info:
                print(f"Warning: Item {i} in perception_info is missing 'box'. Skipping.")
                continue

            box = info["box"]
            label = str(i + 1)
            # Ensure box coordinates are integers
            try:
                x1, y1, x2, y2 = map(int, box)
                # min_wh = min(x2 - x1, y2 - y1) // 4
                # x1 = max(0, x1 - min_wh)
                # y1 = max(0, y1 - min_wh)
                box_width = x2 - x1
                box_height = y2 - y1
                font_scale = 1080 / min(image_width, image_height)
                box_size = min(box_width, box_height)
                font_size = max(20, min(40, int(font_scale * box_size / 6)))
                font = utils.get_font(font_size)

                # Get text dimensions
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_y_offset = text_bbox[1]
                x1 -= text_width * 0.5
                x1 = max(0, x1)
                y1 -= text_height * 0.5
                y1 = max(0, y1)
            except (ValueError, TypeError):
                print(f"Warning: Invalid box coordinates for item {i}: {box}. Skipping.")
                continue

            if y1 >= y2 or x1 >= x2:
                continue

            # Get color for this box
            base_color = colors[i % len(colors)]

            # 1. Draw semi-transparent fill on the overlay
            fill_color = base_color + (int(255 * alpha_percent),)  # Add alpha channel
            overlay_draw.rectangle([(x1, y1), (x2, y2)], fill=fill_color)

            # 2. Draw box outline on the main image
            outline_color = base_color  # Use base RGB color for outline
            draw.rectangle([(x1, y1), (x2, y2)], outline=outline_color, width=outline_thickness)

            # if text_height * 2 > box_height:
            #     continue

            # Place label inside the top-left corner
            text_x = x1 + label_margin
            text_y = y1 + label_margin - text_y_offset

            # Define text area where the background will be analyzed
            text_area = (
                text_x,
                text_y + text_y_offset,
                text_x + text_width,
                text_y + text_y_offset + text_height
            )

            # Get dominant background color in text region
            bg_color = get_dominant_color(image, text_area)

            # Get a contrasting color for the text
            text_color = get_contrasting_color(bg_color)

            # Draw the text
            draw.text((text_x, text_y), label, fill=text_color, font=font)

        # Composite the overlay with the image containing outlines and text
        img_draw = Image.alpha_composite(img_draw, overlay)

        return np.array(img_draw)

    def convert_perception_to_string(self, screen_state: PerceptionInfo) -> str:
        """
        把感知信息转为文本，
        :param perception_info: 感知信息列表
        :return: 格式化后的markdown字符串
        """
        perception_string = ""
        if not screen_state:
            return perception_string
        perception_info = screen_state.perception_info
        if not perception_info:
            return perception_string

        for i, info in enumerate(perception_info):
            text = info.get("text", "?")
            if not text or text.strip() == "":
                text = "?"
            perception_string += f"{i + 1}.<{text}>\n"
        return perception_string

    def convert_perception_to_markdown(self, screen_state: PerceptionInfo) -> str:
        """
        把感知信息转为文本，使用网格布局
        :param perception_info: 感知信息列表
        :return: 格式化后的markdown字符串
        """
        if not screen_state:
            return ""
        perception_info = screen_state.perception_info
        if not perception_info:
            return ""

        # 找出所有元素的外包围框
        min_x = min(info["box"][0] for info in perception_info)
        min_y = min(info["box"][1] for info in perception_info)
        max_x = max(info["box"][2] for info in perception_info)
        max_y = max(info["box"][3] for info in perception_info)

        # 计算包围框的宽度和高度
        width = max_x - min_x
        height = max_y - min_y

        # 决定网格的粒度，这里以短边/16作为单位格子大小
        grid_size = min(width, height) / self.config.detect_grid_num
        grid_size = max(1, grid_size)  # 确保网格大小至少为1

        # 计算网格的行数和列数 +100 可以多预留出一点buffer, 宁可多, 不可少
        num_cols = int(width / grid_size) + 100
        num_rows = int(height / grid_size) + 100

        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]

        text_lens = []
        for ocr_info in screen_state.ocr_infos:
            if ocr_info["text"]:
                text_lens.append((ocr_info["box"][2] - ocr_info["box"][0]) / len(ocr_info["text"]))

        if text_lens:
            spaces_per_grid = grid_size / np.median(text_lens)
            spaces_per_grid = round(spaces_per_grid)
            spaces_per_grid = max(2, spaces_per_grid)
        else:
            spaces_per_grid = 2

        # 将元素放入网格中
        for i, info in enumerate(perception_info):
            x1, y1, x2, y2 = info["box"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # 计算元素在网格中的位置
            grid_row = int((cy - min_y) / grid_size)
            grid_col = int((cx - min_x) / grid_size)

            # 获取文本，如果为空则用"?"代替
            text = info.get("text", "?")
            if not text or text.strip() == "":
                text = "?"

            # 将元素放入计算出的确切位置
            if 0 <= grid_row < num_rows and 0 <= grid_col < num_cols:
                # 如果该位置已有元素，我们尝试找个附近的位置
                if grid[grid_row][grid_col] is not None:
                    # 尝试向右查找空位
                    for c in range(grid_col + 1, num_cols):
                        if grid[grid_row][c] is None:
                            grid_col = c
                            break
                    else:
                        # 如果同一行没有空位，尝试下一行
                        for r in range(grid_row + 1, num_rows):
                            for c in range(num_cols):
                                if grid[r][c] is None:
                                    grid_row = r
                                    grid_col = c
                                    break
                            else:
                                continue
                            break

                # 在确定的位置放置元素
                if grid[grid_row][grid_col] is None:  # 再次检查位置是否可用
                    grid[grid_row][grid_col] = f"[{i + 1}]<{text}>"

        # 将网格转换为字符串，去除每行末尾的空白单元格
        result = []
        empty_line_count = 0  # 用于跟踪连续空行的数量

        for row in grid:
            # 找到该行最后一个非空元素的索引
            last_non_empty = -1
            for i in range(len(row) - 1, -1, -1):
                if row[i] is not None:
                    last_non_empty = i
                    break

            # 如果整行都是空的
            if last_non_empty < 0:
                if empty_line_count == 0:
                    # 第一个空行输出一个换行符
                    result.append("")
                    empty_line_count += 1
                else:
                    # 连续的空行跳过
                    empty_line_count += 1
                    continue
            else:
                # 非空行，重置空行计数器
                empty_line_count = 0

                # 构建行内容
                line_elements = []

                # 收集这一行中所有非空元素
                for i in range(last_non_empty + 1):
                    if row[i] is not None:
                        # 记录元素及其位置
                        line_elements.append((i, row[i]))

                # 构建行字符串，确保元素间至少有两个空格
                if line_elements:
                    line = []

                    # 处理第一个元素的位置
                    first_col, first_elem = line_elements[0]
                    # 添加与左边界的空格
                    if first_col > 0:
                        line.append(" " * (first_col * spaces_per_grid))
                    line.append(first_elem)

                    # 处理剩余元素，确保相邻元素之间至少有两个空格
                    for j in range(1, len(line_elements)):
                        prev_col = line_elements[j - 1][0]
                        curr_col = line_elements[j][0]
                        curr_elem = line_elements[j][1]

                        # 计算基于列位置的间距
                        col_spaces = (curr_col - prev_col - 1) * spaces_per_grid

                        # 确保至少有两个空格的间距
                        spaces = max(2, col_spaces)
                        line.append(" " * spaces)
                        line.append(curr_elem)

                    result.append("".join(line))

        return "\n".join(result)

    def highlight_action(
            self,
            action_name: str,
            action_params: Any,
            resolved_coords: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Draws a visual representation of the action onto the cached screenshot.

        Args:
            action_name: Name of the action (e.g., 'click', 'swipe').
            action_params: The Pydantic model containing action parameters.
            resolved_coords: Dictionary containing resolved coordinates if applicable
                             (e.g., {'pos': (x,y)} or {'start': (x1,y1), 'end': (x2,y2)}).
        """
        if not self.cached_state or not self.cached_state.screenshot_base64:
            logger.warning("Cannot highlight action: Cached state or screenshot missing.")
            return

        try:
            draw_image_pil = utils.base64_to_pil_image(self.cached_state.screenshot_base64).convert("RGBA")
            draw = ImageDraw.Draw(draw_image_pil)
            font = utils.get_font(font_size=min(draw_image_pil.size) // 32)

            # Define colors and sizes (consider making these configurable)
            CLICK_COLOR = "red"
            SWIPE_COLOR = "green"
            DRAG_COLOR = "blue"
            LONG_PRESS_COLOR = "purple"
            TEXT_COLOR = "black"
            BACKGROUND_COLOR = "rgba(255, 255, 255, 180)"  # Semi-transparent white
            RADIUS = 30
            LINE_WIDTH = 10
            ARROW_LENGTH = 60
            ARROW_ANGLE = math.pi / 6  # 30 degrees
            highlight_action = True

            if action_name in ["click", "double_click", "right_click"] and resolved_coords and 'pos' in resolved_coords:
                cx, cy = resolved_coords['pos']
                # Draw a circle/crosshair
                draw.ellipse([(cx - RADIUS, cy - RADIUS), (cx + RADIUS, cy + RADIUS)], outline=CLICK_COLOR,
                             width=LINE_WIDTH)
                draw.line([(cx - RADIUS, cy), (cx + RADIUS, cy)], fill=CLICK_COLOR, width=LINE_WIDTH // 2)
                draw.line([(cx, cy - RADIUS), (cx, cy + RADIUS)], fill=CLICK_COLOR, width=LINE_WIDTH // 2)
                logger.debug(f"Highlighting click at ({cx:.1f}, {cy:.1f})")

            elif action_name == "long_press" and resolved_coords and 'pos' in resolved_coords:
                cx, cy = resolved_coords['pos']
                # Draw a dashed circle or different shape
                # Simple approach: thicker circle
                outer_radius = RADIUS * 1.2
                draw.ellipse([(cx - outer_radius, cy - outer_radius), (cx + outer_radius, cy + outer_radius)],
                             outline=LONG_PRESS_COLOR, width=LINE_WIDTH + 2)
                # Optional: Add duration text
                duration_text = f"{action_params.duration:.1f}s"
                text_bbox = draw.textbbox((0, 0), duration_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_pos = (cx - text_width / 2, cy + outer_radius + 5)
                # Add background for text
                draw.rectangle(
                    (text_pos[0] - 2, text_pos[1] - 2, text_pos[0] + text_width + 2, text_pos[1] + text_height + 2),
                    fill=BACKGROUND_COLOR)
                draw.text(text_pos, duration_text, fill=TEXT_COLOR, font=font)
                logger.debug(f"Highlighting long press at ({cx:.1f}, {cy:.1f}) for {action_params.duration}s")


            elif action_name in ["swipe",
                                 "drag"] and resolved_coords and 'start' in resolved_coords and 'end' in resolved_coords:
                fx, fy = resolved_coords['start']
                tx, ty = resolved_coords['end']
                color = SWIPE_COLOR if action_name == "swipe" else DRAG_COLOR

                # Draw the main line
                draw.line([(fx, fy), (tx, ty)], fill=color, width=LINE_WIDTH)

                # Draw arrowhead
                angle = math.atan2(ty - fy, tx - fx)
                # Point 1
                x1 = tx - ARROW_LENGTH * math.cos(angle - ARROW_ANGLE)
                y1 = ty - ARROW_LENGTH * math.sin(angle - ARROW_ANGLE)
                # Point 2
                x2 = tx - ARROW_LENGTH * math.cos(angle + ARROW_ANGLE)
                y2 = ty - ARROW_LENGTH * math.sin(angle + ARROW_ANGLE)

                draw.polygon([(tx, ty), (x1, y1), (x2, y2)], fill=color)
                logger.debug(f"Highlighting {action_name} from ({fx:.1f}, {fy:.1f}) to ({tx:.1f}, {ty:.1f})")

            elif action_name == "press_key":
                key_name = action_params.key_name
                text = f"Press Key: {key_name}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                # Position at bottom-left, adjust padding as needed
                pos_x = draw_image_pil.width // 2 - text_width // 2
                pos_y = draw_image_pil.height // 2 - text_height // 2
                # Add background for text
                draw.rectangle((pos_x - 5, pos_y - 5, pos_x + text_width + 5, pos_y + text_height + 5),
                               fill=BACKGROUND_COLOR)
                draw.text((pos_x, pos_y), text, fill=TEXT_COLOR, font=font, anchor="lt")
                logger.debug(f"Highlighting press_key: {key_name}")

            elif action_name == "input_text" and resolved_coords and 'pos' in resolved_coords:
                pos_x, pos_y = resolved_coords['pos']
                text_to_input = action_params.text
                text_preview = (text_to_input[:30] + '...') if len(text_to_input) > 30 else text_to_input
                text = f"Input: \"{text_preview}\""
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                draw.rectangle((pos_x - 5, pos_y - 5, pos_x + text_width + 5, pos_y + text_height + 5),
                               fill=BACKGROUND_COLOR)
                draw.text((pos_x, pos_y), text, fill=TEXT_COLOR, font=font, anchor="lt")
                logger.debug(f"Highlighting input_text: {text_preview}")
            else:
                highlight_action = False

            if highlight_action:
                output_screenshot_dir = Path(self.config.screenshot_save_dir)
                output_screenshot_dir.mkdir(parents=True, exist_ok=True)
                original_name = os.path.splitext(os.path.basename(self.cached_state.screenshot_path))[0]
                draw_image_filename = f"{original_name}-highlight-action.png"
                draw_image_path = output_screenshot_dir / draw_image_filename

                draw_image_pil.save(str(draw_image_path))
                self.cached_state.highlight_action_path = str(draw_image_path)
                self.cached_state.highlight_action_base64 = utils.pil_image_to_base64(draw_image_pil)
                logger.info(f"Action '{action_name}' highlighted. Saved base64. Path: {draw_image_path}")

                # 保存动作信息到cached_state
                self.cached_state.last_action_info = {
                    "action_name": action_name,
                    "action_params": action_params.model_dump() if hasattr(action_params,
                                                                           'model_dump') else action_params,
                    "resolved_coords": resolved_coords,
                    "highlight_path": draw_image_path,
                    "screenshot_path": self.cached_state.screenshot_path
                }

        except Exception as e:
            logger.error(f"Failed to highlight action '{action_name}': {e}", exc_info=True)

    def get_perception_info(self, image_bgr, **kwargs) -> PerceptionInfo | None:
        """
        通过视觉获取屏幕的感知信息
        :param screenshot_path:
        :return:
        """
        if self.config.perception_type == "local":
            perception_info = self.perception_model.run_perception(image_bgr, **kwargs)

            # # save ocr_infro
            # if perception_info and hasattr(perception_info, "ocr_infos"):
            #     self._save_ocr_results(
            #         perception_info.ocr_infos,
            #         self.config.ocr_results_path,
            #         kwargs.get('screenshot_path', 'unknown.png')
            #     )
            return perception_info
        else:
            return None

    def _save_ocr_results(self, ocr_results, output_path, screenshot_path):
        """保存OCR结果到统一JSON文件"""
        if not ocr_results:
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ocr_data = {
            "timestamp": datetime.now().isoformat(),
            "screenshot": screenshot_path,
            "results": []
        }

        for result in ocr_results:
            ocr_entry = {
                "text": result['text'],
                "confidence": result.get('score', 1.0),
                "box": result.get('box', [0, 0, 0, 0])
            }
            ocr_data["results"].append(ocr_entry)

        try:
            mode = "r+" if os.path.exists(output_path) else "w"
            with open(output_path, mode, encoding="utf-8") as f:
                if mode == "r+":
                    existing_data = json.load(f)
                    existing_data.append(ocr_data)
                    f.seek(0)
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)
                else:
                    json.dump([ocr_data], f, ensure_ascii=False, indent=4)
            logger.info(f"Saved OCR results for {screenshot_path} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save OCR results: {str(e)}")

    def update_state(self, image_path: Optional[str]=None) -> SystemState:
        """
        获取系统当前的状态 state
        :return:
        """
        # 优先使用传入的图片路径（如有）
        if image_path and os.path.exists(image_path):
            screenshot_path = image_path
        else:
            # 使用本地模式获取截图
            screenshot_path = self.take_screenshot()
        if screenshot_path and os.path.exists(screenshot_path):

            output_dir = Path(self.config.screenshot_save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            screenshot_dir = output_dir
            screenshot_dir.mkdir(parents=True, exist_ok=True)

            original_filename = os.path.basename(screenshot_path)
            new_screenshot_path = screenshot_dir / original_filename
            if screenshot_path != str(new_screenshot_path):
                shutil.copy2(screenshot_path, new_screenshot_path)
                logger.info(f"Copied screenshot to: {new_screenshot_path}")
            screenshot_path = str(new_screenshot_path)



            image_bgr = cv2.imread(screenshot_path)
            image_base64 = utils.image_numpy_to_base64(image_bgr)

            highlight_screenshot_path = os.path.splitext(screenshot_path)[0] + "-highlight.png"
            perception_info = None
            if self.config.use_perception:
                perception_info = self.get_perception_info(image_bgr,
                                                           # ocr_results_path=self.config.ocr_results_path,  # 传递OCR路径
                                                           split_num_x=self.config.detect_split_x,
                                                           split_num_y=self.config.detect_split_y,
                                                           predict_type=self.config.detect_type,
                                                           grid_num=self.config.detect_grid_num,
                                                           detect_imgsz=self.config.detect_imgsz,
                                                           split_text=self.config.ocr_split_text,
                                                           use_ocr=self.config.use_ocr,
                                                           use_icon_detect=self.config.use_icon_detect,
                                                           use_template_match=self.config.use_template_match
                                                           )
            image_draw = image_bgr.copy()

            if self.config.highlight_elements:
                if self.config.highlight_type == "normal":
                    image_draw = self.highlight(image_draw, perception_info)
                    highlight_screenshot_path = os.path.splitext(screenshot_path)[0] + "-highlight.png"

                    cv2.imwrite(highlight_screenshot_path, image_draw)
                    logger.info(
                        f"Highlight screenshot in {self.config.highlight_type} type and save at: {highlight_screenshot_path}")
                elif self.config.highlight_type == "grid":
                    image_draw = utils.add_grid_with_numbers(image_draw,
                                                             grid_num=self.config.highlight_grid_num)
                    highlight_screenshot_path = os.path.splitext(screenshot_path)[0] + "-highlight.png"
                    cv2.imwrite(highlight_screenshot_path, image_draw)
                    logger.info(
                        f"Highlight screenshot in {self.config.highlight_type} type and save at: {highlight_screenshot_path}")
            highlight_image_base64 = utils.image_numpy_to_base64(image_draw)
            if self.config.perception_description_type == "md":
                perception_description = self.convert_perception_to_markdown(screen_state=perception_info)
            else:
                perception_description = self.convert_perception_to_string(screen_state=perception_info)
            current_state = SystemState(
                screenshot_dim=(image_bgr.shape[1], image_bgr.shape[0]),
                screenshot_path=screenshot_path,
                screenshot_base64=image_base64,
                highlight_screenshot_path=highlight_screenshot_path,
                highlight_screenshot_base64=highlight_image_base64,
                perception_infos=copy.deepcopy(perception_info) if perception_info is not None else None,
                perception_description=perception_description
            )
            self.cached_state = copy.deepcopy(current_state)
            return self.cached_state
        else:
            logger.error(f"Screenshot {screenshot_path} is not exists!")
            self.cached_state = SystemState()
            return self.cached_state
