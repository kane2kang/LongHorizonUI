import os
import pdb
import traceback
import uuid
from typing import Type, Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
import logging
import json
from PIL import ImageDraw, ImageFont, Image
import textwrap
import time
import copy
from LonghorizonAgent.controller.registry.views import ActionModel
from LonghorizonAgent.controller.views import ActionResult

from collections import OrderedDict
from pathlib import Path

from ..system.system_context import SystemContext, SystemState
from ..controller.base_controller import BaseController
from ..controller.android_controller import AndroidController
from ..common.llm_provider import LLMProvider
from ..prompts.auto_execution_prompt import AutoExecSystemPrompt, AutoExecAgentPrompt
from ..common import utils

logger = logging.getLogger(__name__)


class ProgressMonitor:
    """
    论文 §2.4 实时进度监控器：
    构建时间状态链 (temporal state chain)，捕获屏幕状态与执行结果，
    检测连续N步无进展时触发回退。
    """

    def __init__(self, stagnation_threshold: int = 3, max_rollbacks: int = 3):
        """
        Args:
            stagnation_threshold: 连续失败/无进展步数阈值，超过则触发回退
            max_rollbacks: 单次任务最大回退次数
        """
        self.stagnation_threshold = stagnation_threshold
        self.max_rollbacks = max_rollbacks
        self.state_chain: List[Dict[str, Any]] = []  # 时间状态链
        self.consecutive_failures: int = 0  # 连续失败计数
        self.rollback_count: int = 0  # 已回退次数
        self.committed_snapshots: List[Dict[str, Any]] = []  # 已提交的快照栈

    def record_step(self, step: int, action_data: Dict, eval_result: str,
                    action_result: 'ActionResult', state: 'SystemState'):
        """
        记录每一步的执行状态到时间状态链。

        Args:
            step: 当前步数
            action_data: 执行的动作数据
            eval_result: LLM 返回的 evaluation_prev_goal 字段
            action_result: 动作执行结果
            state: 当前系统状态
        """
        record = {
            "step": step,
            "action_data": action_data,
            "eval_result": eval_result,
            "action_success": action_result.success if action_result else False,
            "action_error": action_result.error if action_result else None,
            "is_done": action_result.is_done if action_result else False,
            "timestamp": time.time()
        }
        self.state_chain.append(record)

        # 判断是否为失败/停滞
        is_failure = self._is_step_failure(eval_result, action_result)
        if is_failure:
            self.consecutive_failures += 1
            logger.warning(f"[进度监控] 连续失败次数: {self.consecutive_failures}/{self.stagnation_threshold}")
        else:
            self.consecutive_failures = 0

    def _is_step_failure(self, eval_result: str, action_result: 'ActionResult') -> bool:
        """判断当前步骤是否为失败/无进展"""
        # 1. 动作执行本身失败
        if action_result and action_result.error:
            return True
        # 2. LLM 评估上一步为 Failed
        if eval_result and "failed" in eval_result.lower():
            return True
        return False

    def should_rollback(self) -> bool:
        """
        论文 Algorithm 1 第13行：检测是否需要触发回退。
        当连续失败次数达到阈值且还有回退额度时触发。
        """
        if self.consecutive_failures >= self.stagnation_threshold:
            if self.rollback_count < self.max_rollbacks:
                return True
            else:
                logger.warning(f"[进度监控] 已达最大回退次数({self.max_rollbacks})，不再回退")
        return False

    def commit_snapshot(self, step: int, state: 'SystemState',
                        chat_messages: List[Dict[str, Any]]):
        """
        提交成功快照：当一步成功执行后，保存当前状态作为回退点。
        对应论文中 "last committed snapshot (s_{t-1}, p_{t-1})"
        """
        snapshot = {
            "step": step,
            "state": copy.deepcopy(state),
            "chat_messages_len": len(chat_messages),  # 记录消息长度用于回退
            "timestamp": time.time()
        }
        self.committed_snapshots.append(snapshot)
        # 只保留最近5个快照，节省内存
        if len(self.committed_snapshots) > 5:
            self.committed_snapshots.pop(0)
        logger.debug(f"[进度监控] 提交快照: step={step}, 快照栈深度={len(self.committed_snapshots)}")

    def pop_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        弹出最近的已提交快照用于回退。
        对应论文 Algorithm 1 第13行: Rollback(s_{t-1}, p_{t-1})
        """
        if self.committed_snapshots:
            snapshot = self.committed_snapshots.pop()
            self.rollback_count += 1
            self.consecutive_failures = 0  # 回退后重置失败计数
            logger.info(f"[进度监控] 执行回退到 step={snapshot['step']}, "
                        f"已回退次数: {self.rollback_count}/{self.max_rollbacks}")
            return snapshot
        logger.warning("[进度监控] 无可用快照，无法回退")
        return None

    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要信息"""
        total_steps = len(self.state_chain)
        success_steps = sum(1 for r in self.state_chain
                           if not self._is_step_failure(r.get("eval_result", ""),
                                                       type('obj', (), r)() if False else None))
        return {
            "total_steps": total_steps,
            "consecutive_failures": self.consecutive_failures,
            "rollback_count": self.rollback_count,
            "snapshot_depth": len(self.committed_snapshots),
            "stagnation_threshold": self.stagnation_threshold
        }


@dataclass
class AutoExecutionConfig:
    """Configuration for the AutoExecutionAgent."""
    agent_output_dir = "./tmp/agent_outputs"
    max_steps: int = 100  # Default max steps to prevent infinite loops
    llm_temperature: float = 0.4
    max_input_len: int = 50
    keep_last_n_states: int = 2
    # 进度监控与回退配置
    stagnation_threshold: int = 3   # 连续失败N步后触发回退
    max_rollbacks: int = 3          # 单次任务最大回退次数
    enable_fallback_cascade: bool = True  # 是否启用三级降级
    enable_progress_monitor: bool = True  # 是否启用进度监控


class AutoExecutionAgent:
    """
    An agent that automates GUI tasks based on visual input and LLM reasoning.
    It observes the screen, thinks, and acts using a controller.
    """

    def __init__(self,
                 agent_config: AutoExecutionConfig,
                 llm: LLMProvider,
                 controller: BaseController,
                 system_context: SystemContext,
                 system_prompt_class: Type[AutoExecSystemPrompt] = AutoExecSystemPrompt,
                 agent_prompt_class: Type[AutoExecAgentPrompt] = AutoExecAgentPrompt,
                 step_output_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 task_dir = None
                 ):
        self.agent_config = agent_config
        self.llm = llm
        self.controller = controller
        self.system_context = system_context
        self.system_prompt_class = system_prompt_class
        self.agent_prompt_class = agent_prompt_class
        self.chat_messages: List[Dict[str, Any]] = []
        self.ActionModelType: Type[ActionModel] = self.controller.registry.create_action_model()
        self.step_output_callback = step_output_callback

        # 新增
        self.task_dir = Path(task_dir) if task_dir else None
        self.DEFAULT_MAX_RETRIES = 3  # 添加类属性
        self._img_cache = OrderedDict()
        self._max_cache_size = 50  # 可配置参数
        # 新增 action records
        self.action_records = []

        # 论文 §2.4: 进度监控器
        self.progress_monitor = ProgressMonitor(
            stagnation_threshold=agent_config.stagnation_threshold,
            max_rollbacks=agent_config.max_rollbacks
        )

    def _initialize_chat(self, task: str, task_steps: Optional[str], task_infos: Optional[str]):
        """Sets up the initial chat messages with system prompt and task description."""
        self.task = task
        self.task_steps = task_steps
        self.task_infos = task_infos

        self.history = []
        self.chat_messages = []
        self.agent_id = str(uuid.uuid4())
        # raw
        # self.agent_output_dir = os.path.join(self.agent_config.agent_output_dir,
        #                                      f"{self.__class__.__name__}-{self.agent_id}")

        # 获取设备ID
        android_device = self.system_context.config.device_id
        if android_device:
            android_device = android_device.split("_")[-1]
        else:
            android_device = "device"  # 或者给它一个默认值

        # 从task_infos中提取原始目录名
        try:
            input_folder_name = task_infos.split(":")[-1].strip().replace(" ", "_")
        except:
            input_folder_name = f"task_{uuid.uuid4().hex[:6]}"

        # 构建输出路径
        self.agent_output_dir = Path(self.agent_config.agent_output_dir) / \
                                f"{input_folder_name}_test"
        self.stopped = False

        self.system_context.config.screenshot_save_dir = os.path.join(self.agent_output_dir, "screenshots")

        # 1. System Prompt
        available_actions_desc = self.controller.registry.get_prompt_description()
        system_prompt_gen = self.system_prompt_class(available_actions_desc)
        system_prompt = system_prompt_gen.get_system_prompt()
        self.chat_messages = self.llm.add_message("system", system_prompt, self.chat_messages)
        logger.debug("Initialized with System Prompt.")
        # logger.debug(f"System Prompt:\n{system_prompt}") # Can be very long

        # 2. Task Description Prompt (as a user message)
        task_prompt_parts = [f"**Overall Task:**\n{task}"]
        if task_steps:
            task_prompt_parts.append(f"\n**Task Steps (Optional):**\n{task_steps}")
        if task_infos:
            task_prompt_parts.append(f"\n**Additional Information:**\n{task_infos}")
        task_prompt_parts.append("\nPlease begin the task.")
        task_description_prompt = "\n".join(task_prompt_parts)

        self.chat_messages = self.llm.add_message("user", task_description_prompt, self.chat_messages)
        logger.debug(f"Added Task Description Prompt:\n{task_description_prompt}")

        logger.info(f"Chat initialized for task: {task}")


    def _add_to_cache(self, key, value):
        """LRU缓存管理方法"""
        if key in self._img_cache:
            self._img_cache.move_to_end(key)
            return
        if len(self._img_cache) >= self._max_cache_size:
            self._img_cache.popitem(last=False)
        self._img_cache[key] = value

    def stop(self) -> None:
        """Stop the agent"""
        logger.info('⏹️ Agent stopping')
        self.stopped = True

    def _make_history_item(
            self,
            model_output: Dict,
            system_state: SystemState,
            result: ActionResult,
    ) -> None:
        """Create and store history item"""

        history_item = {
            "task_infos": {
                "task": self.task,
                "task_steps": self.task_steps,
                "task_infos": self.task_infos
            },
            "model_output": model_output,
            "system_state": asdict(system_state),
            "action_result": result.model_dump(exclude_none=True)
        }
        self.history.append(history_item)

    def save_history(self):
        """
        Save history to json
        :return:
        """
        history_json_path = os.path.join(self.agent_output_dir, f"{self.agent_id}.json")
        os.makedirs(os.path.dirname(history_json_path), exist_ok=True)
        with open(history_json_path, "w", encoding="utf-8") as fw:
            json.dump(self.history, fw)
            logger.info(f"Save agent history at: {history_json_path}")

    def _add_text_to_image(self, image: "Image.Image", step_text: str, goal_text: str):
        """Adds adaptive step number and goal text overlays to an image.
           Draws background directly before text.
        """
        from PIL import ImageDraw, ImageFont  # Import here

        # Ensure image is RGBA for drawing with transparency
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        draw = ImageDraw.Draw(image)  # Draw directly on the RGBA image
        img_width, img_height = image.size
        # Use config values directly or define them here if not using config
        margin = int(img_height * 0.02)  # Example ratio
        text_bg_opacity = 220  # Example: Higher opacity
        font_size_step = max(10, int(img_height * 0.025))  # Example ratio
        font_size_goal = max(12, int(img_height * 0.03))  # Example ratio

        # --- Step Text (Bottom Left) ---
        step_pos_y = img_height
        try:
            font_step = utils.get_font(font_size_step)
            step_bbox = draw.textbbox((0, 0), step_text, font=font_step, anchor="lt")
            if font_step:
                step_bbox = draw.textbbox((0, 0), step_text, font=font_step, anchor="lt")
                step_width = step_bbox[2] - step_bbox[0]
                step_height = step_bbox[3] - step_bbox[1]
                step_pos_x = margin
                step_pos_y = margin

                bg_coords = (
                    step_pos_x - 2, step_pos_y - 2,
                    step_pos_x + step_width + 2, step_pos_y + step_height + 2
                )
                # --- Draw background DIRECTLY before text ---
                draw.rectangle(bg_coords, fill=(0, 0, 0, text_bg_opacity))
                # --- Draw text ---
                draw.text((step_pos_x, step_pos_y), step_text, font=font_step, fill=(255, 255, 255),
                          anchor="lt")  # White text
        except Exception as e:
            logger.warning(f"Could not draw step text '{step_text}': {e}")

        # --- Goal Text (Bottom Center, with Wrapping) ---
        try:
            font_goal = utils.get_font(font_size_goal)
            if font_goal and goal_text:
                max_text_width = int(img_width * 0.9)  # Example ratio
                avg_char_width = draw.textlength(goal_text[0], font=font_goal)
                wrap_width_chars = max(10, int(max_text_width / (avg_char_width + 1)))
                wrapped_lines = textwrap.wrap(goal_text, width=wrap_width_chars)

                total_text_height = 0
                line_heights = []
                for line in wrapped_lines:
                    line_bbox = draw.textbbox((0, 0), line, font=font_goal, anchor="lt")
                    line_height = line_bbox[3] - line_bbox[1]
                    line_heights.append(line_height)
                    total_text_height += line_height
                total_text_height += max(0, (len(wrapped_lines) - 1)) * (margin // 4)

                current_y = img_height - total_text_height - margin
                # Basic overlap check (adjust if needed)
                # if current_y < step_pos_y + (margin // 2):
                #     pass # Accept overlap for now

                for i, line in enumerate(wrapped_lines):
                    line_width = draw.textlength(line, font=font_goal)
                    line_x = (img_width - line_width) / 2
                    line_y = current_y

                    bg_coords_line = (
                        line_x - 5, line_y - 2,
                        line_x + line_width + 5, line_y + line_heights[i] + 2
                    )
                    # --- Draw background DIRECTLY before text ---
                    draw.rectangle(bg_coords_line, fill=(0, 0, 0, text_bg_opacity))
                    # --- Draw text ---
                    draw.text((line_x, line_y), line, font=font_goal, fill=(255, 255, 255), anchor="lt")  # White text

                    current_y += line_heights[i] + (margin // 4)

        except Exception as e:
            logger.warning(f"Could not draw goal text '{goal_text}': {e}")

        # Return the modified RGBA image
        return image



    """实时翻译的历史信息、执行目标描述等推理信息"""
    def _create_info_panel(self, model_output: dict) -> "Image.Image":

        from PIL import Image, ImageDraw, ImageFont
        import re

        # 翻译映射表
        TRANSLATION_CACHE = {
            "N/A": "暂无信息",
            "click": "点击",
            "launch the app": "启动应用",
            "visible": "可见",
            "application": "应用"
        }

        def translate_with_llm(text: str) -> str:
            """调用大模型进行文本翻译"""
            if not text.strip() or text.strip() in TRANSLATION_CACHE:
                return TRANSLATION_CACHE.get(text.strip(), text)

            try:
                # build prompt
                translate_prompt = (
                    f"将以下技术文档内容精准翻译为简体中文，保留专业术语和数字：\n{text}\n"
                    "翻译要求：\n"
                    "1. 保持原有格式和标点\n"
                    "2. 专业术语不翻译（如NetCap）\n"
                    "3. 如果翻译的结果中有类似索引-X(索引2)请帮我删除它 \n"
                    "4. 输出纯文本不要markdown"
                )

                # invoking LLM interface
                llm_response = self.llm.invoke(
                    [{"role": "user", "content": translate_prompt}],
                    temperature=0.1
                )
                translated = llm_response.strip()
                translated = translated.replace("```", "").replace("翻译结果：", "")

                # 删除类似“indexX”或“索引X”的描述
                translated = re.sub(r'\bindex\d+\b', '', translated)
                translated = re.sub(r'\b索引\d+\b', '', translated)

                return translated

            except Exception as e:
                logger.warning(f"Translation failed: {e}, keeping original text")
                return text

        fields = ['import_contents', 'think', 'next_goal', 'action']
        combined_text = "\n".join([str(model_output.get(field, "")).strip() for field in fields])
        translated_combined = translate_with_llm(combined_text)
        translated_output = {}
        translated_lines = translated_combined.split('\n')

        start_index = 0
        for field in fields:
            field_text = str(model_output.get(field, "")).strip()
            translated_output[field] = "\n".join(
                translated_lines[start_index:start_index + field_text.count('\n') + 1]).strip()
            start_index += field_text.count('\n') + 1

        PANEL_WIDTH = 1080
        PANEL_HEIGHT = 2340
        panel = Image.new('RGB', (PANEL_WIDTH, PANEL_HEIGHT), (255, 255, 255))
        draw = ImageDraw.Draw(panel)

        font_title = utils.get_font_chinese(60)
        font_content = utils.get_font_chinese(48)

        MARGIN_X = 50
        CONTENT_INDENT = 30
        INITIAL_Y = 100
        LINE_SPACING = 35
        SECTION_SPACING = 90
        RIGHT_MARGIN = 50

        current_y = INITIAL_Y
        max_line_width = PANEL_WIDTH - MARGIN_X - CONTENT_INDENT - RIGHT_MARGIN

        fields_and_labels = [
            ('import_contents', '历史操作与场景描述分析', (0, 100, 200)),
            ('think', '执行推理', (200, 100, 0)),
            ('next_goal', '执行目标', (160, 0, 200)),
            ('action', '执行操作', (0, 150, 0)),
        ]

        for field_key, label, color in fields_and_labels:
            title_font_size = 60
            font_title = utils.get_font_chinese(title_font_size)
            title_text = f"{label}："

            while draw.textlength(title_text, font=font_title) > max_line_width and title_font_size >= 48:
                title_font_size -= 2
                font_title = utils.get_font_chinese(title_font_size)

            draw.text((MARGIN_X, current_y), title_text, fill=color, font=font_title)
            current_y += font_title.size + LINE_SPACING

            # get翻译内容
            content = translated_output.get(field_key, "暂无信息")

            # 换行算法
            def split_chinese_lines(text: str, font) -> list:
                lines = []
                current_line = []
                current_width = 0

                for char in text:
                    char_width = draw.textlength(char, font=font)
                    if current_width + char_width > max_line_width:
                        lines.append(''.join(current_line))
                        current_line = [char]
                        current_width = char_width
                    else:
                        current_line.append(char)
                        current_width += char_width

                    # 中文标点换行
                    if char in '。！？；’”）' and current_width > max_line_width * 0.8:
                        lines.append(''.join(current_line))
                        current_line = []
                        current_width = 0

                if current_line:
                    lines.append(''.join(current_line))
                return lines

            wrapped_lines = split_chinese_lines(content, font_content)
            max_lines = 7
            for i, line in enumerate(wrapped_lines[:max_lines]):
                if current_y + font_content.size > PANEL_HEIGHT - 100:
                    draw.text((MARGIN_X + CONTENT_INDENT, current_y), "...",
                              fill=(150, 150, 150), font=font_content)
                    break

                draw.text((MARGIN_X + CONTENT_INDENT, current_y),
                          line, fill=(50, 50, 50), font=font_content)
                current_y += font_content.size + LINE_SPACING

            current_y += SECTION_SPACING

        return panel

    def create_gif(self):
        """
        Create a GIF from the agent's history with adaptive text.
        """
        logger.info("Creating GIF from history...")
        gif_path = os.path.join(self.agent_output_dir, f"{self.agent_id}.gif")
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)

        frames = []
        target_dims: Optional[Tuple[int, int]] = None

        try:
            from PIL import Image, ImageDraw, ImageFont  # Import Pillow components
        except ImportError:
            logger.error("Pillow library not found. Cannot create GIF. Please install Pillow (`pip install Pillow`).")
            return

        # --- 1. Determine Target Dimensions from first step's image ---
        # (This part remains the same as before)
        for i, item in enumerate(self.history):
            state = item.get("system_state", {})
            b64_data = state.get("screenshot_base64") or \
                       state.get("highlight_screenshot_base64") or \
                       state.get("highlight_action_base64")
            if b64_data:
                img_stream = utils.decode_image(b64_data)
                if img_stream:
                    try:
                        with Image.open(img_stream) as img:
                            target_dims = img.size
                            logger.info(
                                f"GIF target dimensions set to {target_dims} based on first image found in step {i + 1}.")
                            break
                    except Exception as e:
                        logger.warning(f"Could not open first image in step {i + 1} to get dimensions: {e}")
        if not target_dims:
            logger.warning("Could not determine target dimensions. Using default 600x800.")
            return

        # # --- 2. Create demo ---
        # try:
        #     GIF_WIDTH = 2160
        #     GIF_HEIGHT = 2340
        #     title_img = Image.new('RGB', (GIF_WIDTH, GIF_HEIGHT), (255, 255, 255))
        #     draw = ImageDraw.Draw(title_img)
        #
        #     # 动态字体配置
        #     base_font_size = 94
        #     font_task = utils.get_font_chinese(base_font_size)
        #
        #     if font_task and self.task:
        #         # 高级排版参数
        #         max_line_chars = 20  # 每行最多字符数
        #         line_spacing = 0.3  # 行间距系数
        #         side_margin = 150  # 两侧安全边距
        #
        #         wrapped_lines = []
        #         current_line = []
        #         for char in self.task:
        #             if len(current_line) >= max_line_chars:
        #                 wrapped_lines.append(''.join(current_line))
        #                 current_line = [char]
        #             else:
        #                 current_line.append(char)
        #         if current_line:
        #             wrapped_lines.append(''.join(current_line))
        #
        #         line_height = int(base_font_size * (1 + line_spacing))
        #         total_height = len(wrapped_lines) * line_height
        #
        #         y_position = (GIF_HEIGHT - total_height) // 2
        #
        #         for line in wrapped_lines:
        #             line_width = draw.textlength(line, font=font_task)
        #             actual_font_size = base_font_size
        #             current_font = font_task
        #             max_allow_width = GIF_WIDTH - 2 * side_margin
        #             while line_width > max_allow_width and actual_font_size > 30:
        #                 actual_font_size -= 2
        #                 current_font = utils.get_font_chinese(actual_font_size)
        #                 line_width = draw.textlength(line, font=current_font)
        #                 line_height = int(actual_font_size * (1 + line_spacing))
        #             x_position = (GIF_WIDTH - line_width) // 2
        #
        #             # draw text
        #             draw.text(
        #                 (x_position, y_position),
        #                 line,
        #                 fill=(0, 0, 0),
        #                 font=current_font,
        #                 stroke_width=1,
        #                 stroke_fill=(150, 150, 150)
        #             )
        #
        #             y_position += line_height  # 换行
        #
        #     draw.line([(GIF_WIDTH // 4, GIF_HEIGHT - 100), (GIF_WIDTH * 3 // 4, GIF_HEIGHT - 100)],
        #               fill=(200, 200, 200), width=2)
        #
        #     frames.append(title_img)
        #     logger.info(f"Created title frame with {len(wrapped_lines)} lines")
        # except Exception as e:
        #     logger.error(f"Failed to create title frame: {e}", exc_info=True)
        #     # 创建保底帧
        #     error_frame = Image.new('RGB', (GIF_WIDTH, GIF_HEIGHT), (255, 255, 225))
        #     draw = ImageDraw.Draw(error_frame)
        #     draw.text((100, 1000), "标题生成失败", fill=(255, 0, 0), font=utils.get_font_chinese(72))
        #     frames.append(error_frame)
        #
        # # --- 3. Process History Frames ---
        # for i, item in enumerate(self.history):
        #     step_num = i + 1
        #     state = item.get("system_state", {})
        #     model_output = item.get("model_output", {})
        #     step_text = f"Step {step_num}"
        #
        #     # 固定尺寸
        #     GIF_WIDTH = 2160
        #     GIF_HEIGHT = 2340
        #
        #     # 处理所有三种图像类型
        #     for img_type in ["screenshot", "highlight_screenshot", "highlight_action"]:
        #         img_b64 = state.get(f"{img_type}_base64")
        #         if not img_b64:
        #             continue
        #
        #         try:
        #             img_stream = utils.decode_image(img_b64)
        #             with Image.open(img_stream) as original_img:
        #                 img = original_img.convert("RGB")
        #                 original_width, original_height = img.size
        #
        #                 # 横屏处理（旋转+缩
        #                 if original_width > original_height:  # 横屏检测
        #
        #                     img = img.rotate(-90, expand=True)
        #
        #                     original_width, original_height = img.size
        #
        #                 # 缩放图像到左侧区域1080x2340
        #                 scale_ratio = min(1080 / original_width, 2340 / original_height)
        #                 new_size = (
        #                     int(original_width * scale_ratio),
        #                     int(original_height * scale_ratio)
        #                 )
        #                 resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        #
        #                 # create text frame
        #                 left_panel = Image.new('RGB', (1080, 2340), (255, 255, 255))
        #                 paste_x = (1080 - new_size[0]) // 2
        #                 paste_y = (2340 - new_size[1]) // 2
        #                 left_panel.paste(resized_img, (paste_x, paste_y))
        #                 try:
        #                     info_panel = self._create_info_panel(model_output)
        #                 except Exception as e:
        #                     logger.error(f"Info panel error: {str(e)}")
        #                     info_panel = Image.new('RGB', (1080, 2340), (255, 255, 255))
        #
        #                 # compose frame
        #                 combined = Image.new('RGB', (GIF_WIDTH, GIF_HEIGHT), (255, 255, 255))
        #                 combined.paste(left_panel, (0, 0))
        #                 combined.paste(info_panel, (1080, 0))
        #
        #                 draw = ImageDraw.Draw(combined)
        #                 font_size = 72
        #                 current_font = utils.get_font_chinese(font_size)
        #
        #                 # 将step_text放置在左上角
        #                 draw.text((100, 100), step_text, fill=(0, 0, 0), font=current_font)
        #                 draw.line([(1080, 0), (1080, 2340)], fill=(200, 200, 200), width=2)
        #
        #                 frames.append(combined)
        #                 logger.info(f"Added {img_type} frame ({combined.size})")
        #
        #         except Exception as e:
        #             logger.error(f"Frame processing failed: {str(e)[:100]}")
        #             # 生成错误帧（带错误信息）
        #             error_frame = Image.new('RGB', (GIF_WIDTH, GIF_HEIGHT), (255, 230, 230))
        #             draw = ImageDraw.Draw(error_frame)
        #             text = f"Step {step_num} Error: {str(e)[:100]}"
        #             draw.text((100, 1000), text, fill=(255, 0, 0), font=utils.get_font_chinese(60))
        #             frames.append(error_frame)
        #
        # # --- 4. Save GIF ---
        # if len(frames) > 1:
        #     try:
        #         final_frames = []
        #         for frame in frames:
        #             if frame.size != (GIF_WIDTH, GIF_HEIGHT):
        #                 resized = frame.resize((GIF_WIDTH, GIF_HEIGHT), Image.Resampling.LANCZOS)
        #                 final_frames.append(resized)
        #             else:
        #                 final_frames.append(frame)
        #
        #         final_frames[0].save(gif_path, save_all=True,append_images=final_frames[1:],
        #             duration=500,  # 1.5秒/帧
        #             loop=0,
        #             quality=100,
        #             subsampling=0
        #         )
        #         logger.success(f"GIF saved to {gif_path}")
        #     except Exception as e:
        #         logger.error(f"Final save failed: {str(e)}")
        # else:
        #     logger.warning("No frames to save")

        # --- 2. Create Title Frame ---
        try:
            gif_background_color = (255, 255, 255)
            title_img = Image.new('RGB', target_dims, color=gif_background_color)
            draw = ImageDraw.Draw(title_img)
            # Calculate adaptive task font size
            font_size_task = max(14, int(target_dims[1] * 0.04))
            font_task = utils.get_font(font_size_task)

            if font_task and self.task:
                # Simple centering for title
                max_title_width = int(target_dims[0] * 0.8)  # Max width 90%
                avg_char_width_task = draw.textlength(self.task[0], font=font_task)
                wrap_width_chars_task = int(max_title_width / (avg_char_width_task + 1))
                wrapped_task_lines = textwrap.wrap(self.task, width=wrap_width_chars_task)

                total_task_height = 0
                task_line_heights = []
                for line in wrapped_task_lines:
                    line_bbox = draw.textbbox((0, 0), line, font=font_task)
                    line_height = line_bbox[3] - line_bbox[1]
                    task_line_heights.append(line_height)
                    total_task_height += line_height
                total_task_height += max(0, (len(wrapped_task_lines) - 1)) * (font_size_task // 3)  # Line spacing

                current_task_y = (target_dims[1] - total_task_height) / 2  # Center vertically

                for i, line in enumerate(wrapped_task_lines):
                    line_width = draw.textlength(line, font=font_task)
                    line_x = (target_dims[0] - line_width) / 2  # Center horizontally
                    draw.text((line_x, current_task_y), line, font=font_task, fill=(0, 0, 0), anchor="lt")
                    current_task_y += task_line_heights[i] + (font_size_task // 3)

            frames.append(title_img)
            logger.debug("Created title frame.")
        except Exception as e:
            logger.error(f"Failed to create title frame: {e}", exc_info=True)

        # --- 3. Process History Frames ---
        # (Image loading, resizing/rotating logic remains the same as before)
        for i, item in enumerate(self.history):
            step_num = i + 1
            state = item.get("system_state", {})
            model_output = item.get("model_output", {})
            goal_text = model_output.get("next_goal", "")
            step_text = f"Step {step_num}"

            images_to_process = [
                state.get("screenshot_base64"),
                state.get("highlight_screenshot_base64"),
                state.get("highlight_action_base64")
            ]

            for img_idx, img_b64 in enumerate(images_to_process):  # Use enumerate index for logging
                if not img_b64: continue
                img_stream = utils.decode_image(img_b64)
                if not img_stream: continue

                try:
                    with Image.open(img_stream) as img:
                        img = img.convert("RGBA")  # Start with RGBA
                        current_dims = img.size
                        processed_img = img

                        # --- Dimension Handling (same as before) ---
                        if current_dims != target_dims:
                            if current_dims == (target_dims[1], target_dims[0]):
                                processed_img = img.rotate(-90, expand=True)
                            else:
                                ratio = min(target_dims[0] / current_dims[0], target_dims[1] / current_dims[1])
                                new_size = (int(current_dims[0] * ratio), int(current_dims[1] * ratio))
                                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                                processed_img = Image.new('RGBA', target_dims, (255, 255, 255, 255))
                                paste_x = (target_dims[0] - new_size[0]) // 2
                                paste_y = (target_dims[1] - new_size[1]) // 2
                                processed_img.paste(resized_img, (paste_x, paste_y),
                                                    resized_img if resized_img.mode == 'RGBA' else None)  # Use mask if pasting RGBA

                        # --- Add Text Overlays ---
                        # The text addition now happens on the potentially resized/padded RGBA image
                        img_with_text = self._add_text_to_image(processed_img.copy(), step_text,
                                                                goal_text)  # Pass a copy

                        # --- Convert final frame to RGB for GIF ---
                        final_rgb_frame = Image.new("RGB", img_with_text.size, gif_background_color)
                        final_rgb_frame.paste(img_with_text, mask=img_with_text.split()[3])  # Paste using alpha

                        frames.append(final_rgb_frame)
                        logger.debug(f"Added frame for step {step_num} (image type {img_idx + 1})")

                except Exception as e:
                    logger.error(f"Step {step_num}: Failed to process image type {img_idx + 1}: {e}", exc_info=True)

        # --- 4. Save GIF ---
        # (Saving logic remains the same as before)
        if len(frames) > 1:
            try:
                logger.info(f"Saving GIF with {len(frames)} frames to {gif_path}...")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=[2000] + [1000] * (
                            len(frames) - 1),
                    loop=0
                )
                logger.info(f"GIF saved successfully to {gif_path}")
            except Exception as e:
                logger.error(f"Failed to save GIF: {e}", exc_info=True)
        elif len(frames) == 1:
            logger.warning("Only title frame was generated. Skipping GIF save.")
        else:
            logger.warning("No frames were generated. Skipping GIF save.")




    def manage_memory(self):
        """
        Keeps the chat history concise by retaining only the system prompt,
        the initial task prompt, and the messages related to the last N image states.
        """
        if len(self.chat_messages) <= 2:  # Only system + task prompt
            return

        system_prompt = self.chat_messages[0]
        task_prompt = self.chat_messages[1]
        history = self.chat_messages[2:]

        i = len(history) - 1
        remove_cnt = 0
        while i >= 0:
            msg = history[i]
            if msg.get("role") == "user" and isinstance(msg.get("content"), list) and any(
                    "image_url" in item for item in msg["content"] if isinstance(item, dict)):
                remove_cnt += 1
            if remove_cnt == abs(self.agent_config.keep_last_n_states + 1):
                history.pop(i)
                break
            i -= 1
        self.chat_messages = [system_prompt, task_prompt] + history[-self.agent_config.max_input_len:]

    def _raise_if_stopped(self) -> None:
        """Utility function that raises an InterruptedError if the agent is stopped or paused."""
        if self.stopped:
            logger.warning('Agent Stop!')
            raise InterruptedError

    def _verify_action(self, eval_prev_goal: str, action_result: ActionResult) -> bool:
        """
        论文 §2.4 公式3: Post-execution Verify
        v_t = Verify_MLLM(s_t, a, p_t, I_{t+1}) ∈ {0, 1}

        基于 LLM 返回的 evaluation_prev_goal 字段和动作执行结果
        进行程序化验证，判断动作是否成功。

        Args:
            eval_prev_goal: LLM 输出的 evaluation_prev_goal 字段
            action_result: 动作执行的 ActionResult

        Returns:
            True 表示验证通过 (v_t=1)，False 表示验证失败 (v_t=0)
        """
        # 1. 动作执行本身有错误 → 验证失败
        if action_result and action_result.error:
            logger.info(f"[执行后验证] v_t=0: 动作执行报错 - {action_result.error}")
            return False

        # 2. LLM 评估上一步为 Failed → 验证失败
        if eval_prev_goal:
            eval_lower = eval_prev_goal.lower().strip()
            if eval_lower.startswith("failed") or "failed" in eval_lower.split("-")[0].strip().lower():
                logger.info(f"[执行后验证] v_t=0: LLM 评估上一步失败 - {eval_prev_goal}")
                return False

        # 3. 其他情况视为成功
        logger.debug(f"[执行后验证] v_t=1: 验证通过")
        return True

    def _rollback(self) -> bool:
        """
        论文 Algorithm 1 第13行: Rollback(s_{t-1}, p_{t-1})
        回退到上一个已提交的成功快照状态。

        Returns:
            True 表示回退成功，False 表示无法回退
        """
        snapshot = self.progress_monitor.pop_snapshot()
        if snapshot is None:
            logger.warning("[回退] 无可用快照，回退失败")
            return False

        rollback_step = snapshot["step"]
        rollback_state = snapshot["state"]
        msg_len = snapshot["chat_messages_len"]

        # 恢复系统状态缓存
        self.system_context.cached_state = copy.deepcopy(rollback_state)

        # 截断聊天历史到回退点
        if msg_len > 0 and msg_len < len(self.chat_messages):
            self.chat_messages = self.chat_messages[:msg_len]

        logger.info(f"[回退] 已回退到 step={rollback_step}，"
                    f"聊天历史截断为 {len(self.chat_messages)} 条消息，"
                    f"已回退次数: {self.progress_monitor.rollback_count}/{self.progress_monitor.max_rollbacks}")
        return True

    def step(self, current_step: int, prev_action_result: Optional[ActionResult]) -> ActionResult:
        """
        Performs a single step: Update state, prepare prompt, manage memory, call LLM, parse, execute action.

        Args:
            current_step: The current step number.
            prev_action_result: String summary of the previous action's execution result.

        Returns:
            A tuple containing:
            - The ActionResult from the executed action (or None if a pre-action error occurred).
            - An optional string summarizing the execution result for the *next* step's prompt.
        """
        logger.info(f"--- Agent Step {current_step} ---")
        t0 = time.time()
        self._raise_if_stopped()
        # 1. Update State (Get current screen)
        try:
            logger.debug("Updating system state...")
            current_state = self.system_context.update_state()
            if not current_state or not current_state.highlight_screenshot_base64:
                logger.error("Failed to update system state or get screenshot.")
                error_result = ActionResult(success=False, is_done=True, error="Failed to get current screen state.")
                return error_result
            logger.debug(f"State updated. Screenshot dimensions: {current_state.screenshot_dim}")
        except Exception as e:
            logger.error(f"Failed during system state update: {e}", exc_info=True)
            error_result = ActionResult(success=False, is_done=True, error=f"System state update failed: {e}")
            return error_result
        if self.step_output_callback:
            step_data = {
                "step": current_step,
                "phase": "observation",
                "screenshot": current_state.highlight_screenshot_base64,
            }
            self.step_output_callback(step_data)
        self._raise_if_stopped()
        # 2. Prepare User Prompt for this step
        agent_prompt_gen = self.agent_prompt_class()
        user_prompt = agent_prompt_gen.get_agent_prompt(
            current_step=current_step
        )
        # Add the prompt text and the current screenshot (base64)
        self.chat_messages = self.llm.add_message(
            "user",
            user_prompt,
            self.chat_messages,
            [current_state.highlight_screenshot_base64]  # Add the current image
        )
        logger.debug("Prepared user prompt with current screenshot.")

        # 3. Manage Memory (Prune history *before* LLM call)
        self.manage_memory()
        self._raise_if_stopped()
        # 4. Call LLM
        try:
            logger.debug(f"Sending {len(self.chat_messages)} messages to LLM.")
            llm_response_text = self.llm.invoke(
                self.chat_messages,
                temperature=self.agent_config.llm_temperature
            )
            logger.debug(f"LLM Raw Response:\n{llm_response_text}")  # Log raw for debugging
            # Add assistant response *after* potential parsing errors handled
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            error_result = ActionResult(success=False, is_done=True, error=f"LLM invocation failed: {e}")
            # Don't add assistant message if invoke failed
            return error_result
        self._raise_if_stopped()
        # 5. Parse LLM Response
        try:
            llm_response_text_cleaned = llm_response_text.split("```json")[1].split("```")[0]
            # Ensure it's valid JSON before adding to history
            parsed_response = json.loads(llm_response_text_cleaned)
            action_data = parsed_response.get("action")

            # Add the valid assistant response to chat history *now*
            self.chat_messages = self.llm.add_message("assistant", llm_response_text,
                                                      self.chat_messages)  # Store original response with formatting

            think_process = parsed_response.get("think", "N/A")
            next_goal = parsed_response.get("next_goal", "N/A")
            logger.info(f"LLM Decision:\n"
                        f"  - Eval Prev Goal: {parsed_response.get('evaluation_prev_goal', 'N/A')}\n"
                        f"  - Important Contents: {parsed_response.get('import_contents', 'N/A')}\n"
                        f"  - Thought: {think_process}\n"
                        f"  - Next Goal: {next_goal}\n"
                        f"  - Action: {action_data}")

            if not action_data or not isinstance(action_data, dict) or len(action_data) != 1:
                raise ValueError("Invalid 'action' format in LLM response.")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response or invalid format: {e}\nResponse: {llm_response_text}",
                         exc_info=True)
            # Add the problematic response to history for context before failing
            self.chat_messages = self.llm.add_message("assistant", llm_response_text,
                                                      self.chat_messages)  # Store faulty response
            error_result = ActionResult(success=False, is_done=True, error=f"LLM response parsing failed: {e}")
            return error_result
        self._raise_if_stopped()

        # === 论文 §2.4: Post-execution Verify — 验证上一步执行结果 ===
        eval_prev_goal = parsed_response.get('evaluation_prev_goal', '')
        if prev_action_result and current_step > 1:
            verify_result = self._verify_action(eval_prev_goal, prev_action_result)
            # 记录到进度监控器
            if self.agent_config.enable_progress_monitor:
                self.progress_monitor.record_step(
                    step=current_step - 1,
                    action_data={},
                    eval_result=eval_prev_goal,
                    action_result=prev_action_result,
                    state=current_state
                )

        # 6. Execute Action (论文 Algorithm 1: 三级降级执行)
        try:
            # 判断是否启用三级降级
            use_fallback = (self.agent_config.enable_fallback_cascade
                           and isinstance(self.controller, AndroidController))

            if use_fallback:
                # 论文 Algorithm 1: 按 index → relative → absolute+ε 优先级尝试
                encodings = self.controller.get_fallback_encodings(action_data)
                action_result = None
                used_encoding = None

                for enc in encodings:
                    # 尝试用当前编码解析坐标
                    resolved = self.controller._try_encoding(action_data, enc, self.system_context)
                    if resolved is not None:
                        logger.info(f"[三级降级] 尝试编码 '{enc}'，解析坐标: {resolved}")
                        # 使用原始方式执行动作
                        action_model_instance = self.ActionModelType(**action_data)
                        action_result = self.controller.act(action_model_instance, self.system_context)

                        # 检查执行结果
                        if action_result and not action_result.error:
                            used_encoding = enc
                            logger.info(f"[三级降级] 编码 '{enc}' 执行成功")
                            break
                        else:
                            logger.warning(f"[三级降级] 编码 '{enc}' 执行失败: "
                                         f"{action_result.error if action_result else 'unknown'}")
                            continue  # 降级到下一种编码

                if action_result is None:
                    # 所有编码都失败，使用原始方式执行
                    logger.warning("[三级降级] 所有编码均失败，回退到默认执行")
                    action_model_instance = self.ActionModelType(**action_data)
                    action_result = self.controller.act(action_model_instance, self.system_context)
            else:
                # 不启用降级，使用原始执行方式
                action_model_instance = self.ActionModelType(**action_data)
                logger.info(f"Executing action: {action_model_instance.model_dump(exclude_none=True)}")
                action_result = self.controller.act(action_model_instance, self.system_context)

            # === 新增：收集动作信息（直接使用highlight_action中存储的信息） ===
            if hasattr(current_state, 'last_action_info') and current_state.last_action_info:
                action_record = {
                    "step": current_step,
                    "action_type": current_state.last_action_info.get("action_name", "unknown"),
                    "action_params": current_state.last_action_info.get("action_params", {}),
                    "resolved_coords": current_state.last_action_info.get("resolved_coords", {}),
                    "screenshot_path": current_state.last_action_info.get("screenshot_path", ""),
                    "highlight_path": current_state.last_action_info.get("highlight_path", ""),
                    "timestamp": time.time()
                }
                self.action_records.append(action_record)
                logger.debug(f"Recorded action for step {current_step}: {action_record}")
                current_state.last_action_info = None
            # ======================================================

            action_ret_str = ""
            if action_result.include_in_memory:
                action_ret_str = f"\n**Previous Action Result:**\n"
                if action_result.extracted_content:
                    action_ret_str += f'\nExtracted Content: {action_result.extracted_content}\n'
                if action_result.error:
                    # only use last line of error
                    error = action_result.error.split('\n')[-1]
                    action_ret_str += f'\nError: {error}\n'
                if current_state.highlight_action_base64:
                    action_ret_str += (f"\nHighlighted action screenshot will visually indicate the *exact location* of your last action (e.g., clicks/presses marked with a circle/X, swipes with an arrow). "
                                       f"This image helps you assess if your previous operation landed where intended and allows for precise adjustments if necessary.\n")
            if action_ret_str:
                self.chat_messages = self.llm.add_message("user", action_ret_str, self.chat_messages,
                                                          [
                                                              current_state.highlight_action_base64] if current_state.highlight_action_base64 else [])
            t1 = time.time()
            logger.info(f"Step Time: {t1 - t0}")
            if self.step_output_callback:
                step_data = {
                    "step": current_step,
                    "step_time": t1 - t0,
                    "phase": "action",
                    "screenshot": current_state.highlight_action_base64,
                    "model_output": parsed_response,
                    "action_result": action_result.model_dump(exclude_none=True)
                }
                self.step_output_callback(step_data)
            # always wait a sec
            time.sleep(2.0)
            self._make_history_item(model_output=parsed_response, result=action_result, system_state=current_state)

            # === 论文 §2.4: 提交成功快照到进度监控器 ===
            if self.agent_config.enable_progress_monitor:
                if action_result and not action_result.error and action_result.success is not False:
                    # 执行成功，提交快照 (last committed snapshot)
                    self.progress_monitor.commit_snapshot(
                        step=current_step,
                        state=current_state,
                        chat_messages=self.chat_messages
                    )

            # Prepare feedback string for the *next* step
            next_step_feedback = f"Action: {json.dumps(action_data)}, Execution Result: Success={action_result.success}"
            if action_result.error:
                next_step_feedback += f", Error='{action_result.error}'"
            # Note: We don't include extracted_content here unless specifically requested

            if not action_result.success and action_result.error:
                logger.warning(f"Action execution failed: {action_result.error}")
            if action_result.is_done:
                logger.info("Task marked as done by the action result.")
                # Even if done, return the result and feedback
                return action_result

        except Exception as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            error_result = ActionResult(success=False, is_done=True, error=f"Action execution failed: {e}")
            return error_result

        # 7. Return result for this step
        return action_result

    def run(self, task: str,
            task_steps: Optional[str] = "",
            task_infos: Optional[str] = "",
            max_steps: Optional[int] = None
            ) -> str | None:
        """
        Runs the agent to complete the given task.
        """
        logger.info(f"Starting agent run for task: '{task}'")
        if max_steps is None:
            max_steps = self.agent_config.max_steps

        # 1. Initialize Chat History (System Prompt + Task Description)
        try:
            self._initialize_chat(task, task_steps, task_infos)
        except ValueError as e:
            logger.error(f"Failed to initialize chat: {e}", exc_info=True)
            return None
        # 2. Execution Loop
        last_action_result: Optional[ActionResult] = None

        try:
            for i in range(max_steps):
                step_count = i + 1
                self._raise_if_stopped()

                # === 论文 §2.4: 进度监控 — 检测停滞并触发回退 ===
                if self.agent_config.enable_progress_monitor and self.progress_monitor.should_rollback():
                    logger.warning(f"[进度监控] 检测到连续 {self.progress_monitor.consecutive_failures} 步失败/停滞，"
                                   f"触发回退 (第 {self.progress_monitor.rollback_count + 1} 次)")
                    rollback_success = self._rollback()
                    if rollback_success:
                        # 回退成功后记录事件
                        logger.info(f"[进度监控] 回退成功，从 step {step_count} 继续执行")
                        # 重置上一步结果，让 agent 重新观察当前状态
                        last_action_result = None
                        continue
                    else:
                        logger.warning("[进度监控] 回退失败，继续正常执行")

                # Pass feedback from the *previous* step's execution
                current_action_result = self.step(
                    current_step=step_count,
                    prev_action_result=last_action_result
                )

                last_action_result = current_action_result
                if last_action_result and last_action_result.is_done:
                    if last_action_result.success:
                        logger.info(f"Task completed successfully after {step_count} steps.")
                    else:
                        logger.warning(
                            f"Agent stopped on step {step_count}. Task marked done by action, but action reported non-success")
                    break
            else:
                logger.warning(f"Failed to complete task in maximum steps: {max_steps}")
        finally:
            self.create_gif()
            self.save_history()
            # 保存acton 信息
            if hasattr(self, "action_records") and self.action_records:
                records_path = os.path.join(self.agent_output_dir, "action_records.json")
                with open(records_path, "w") as f:
                    json.dump(self.action_records, f, indent=2, default=str)
                logger.info(f"Saved action records to: {records_path}")

            # 保存进度监控信息
            if self.agent_config.enable_progress_monitor:
                monitor_path = os.path.join(self.agent_output_dir, "progress_monitor.json")
                monitor_data = {
                    "progress_summary": self.progress_monitor.get_progress_summary(),
                    "state_chain": self.progress_monitor.state_chain,
                }
                with open(monitor_path, "w", encoding="utf-8") as f:
                    json.dump(monitor_data, f, indent=2, default=str, ensure_ascii=False)
                logger.info(f"Saved progress monitor data to: {monitor_path}")

            return self.agent_output_dir