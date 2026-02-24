import time
import random
import logging
import traceback
import copy
from typing import Union, Tuple, Optional, Literal, List, Dict, Any

from .base_controller import BaseController
from .views import ActionResult, ClickAction, PositionType, LongPressAction, SwipeAction, \
    InputTextAction, PressKeyAction, DragAction
from ..system.android_context import AndroidContext

logger = logging.getLogger(__name__)


class AndroidController(BaseController):

    def _resolve_position_to_coords(self, position_param: PositionType, system_context: AndroidContext) -> Tuple[
        float, float]:
        """
        Helper function to resolve a position parameter (int index or tuple coords)
        into absolute (x, y) float coordinates.
        Raises ValueError or IndexError if resolution fails.
        """
        cache_state = system_context.cached_state  # Assuming cached_state exists and has perception_infos
        if not cache_state or not hasattr(cache_state, 'perception_infos') or not cache_state.perception_infos:
            raise ValueError("Cached perception info is missing or empty.")

        if isinstance(position_param, int):
            index = position_param - 1  # User sees 1-based index
            if not (0 <= index < len(cache_state.perception_infos.perception_info)):
                raise IndexError(
                    f"Index {position_param} is out of bounds for perception_infos (length: {len(cache_state.perception_infos.perception_info)}).")
            try:
                box = cache_state.perception_infos.perception_info[index]["box"]  # Expecting [x1, y1, x2, y2]
                if len(box) != 4:
                    raise ValueError(f"Box data for index {position_param} has incorrect format: {box}")
                x1, y1, x2, y2 = box
                # Calculate center coordinates as floats
                cx = float(x1 + x2) / 2.0
                cy = float(y1 + y2) / 2.0
                return cx, cy
            except KeyError:
                raise ValueError(f"Key 'box' not found in perception_infos for index {position_param}.")
            except TypeError as e:
                raise ValueError(f"Error processing box data for index {position_param}: {e}")

        elif (isinstance(position_param, tuple) or isinstance(position_param, list)) and len(position_param) == 2:
            try:
                # Ensure coordinates are floats
                cx = float(position_param[0])
                cy = float(position_param[1])
                # Need to do Unormalize
                image_w, image_h = cache_state.screenshot_dim
                cx = cx / 1000 * image_w
                cy = cy / 1000 * image_h
                return cx, cy
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid coordinate tuple format: {position_param}. Error: {e}")
        elif (isinstance(position_param, tuple) or isinstance(position_param, list)) and len(position_param) == 3:
            index = position_param[0] - 1  # User sees 1-based index
            rx = min(max(0, float(position_param[1])), 1.0)
            ry = min(max(0, float(position_param[2])), 1.0)
            if not (0 <= index < len(cache_state.perception_infos.perception_info)):
                raise IndexError(
                    f"Index {position_param} is out of bounds for perception_infos (length: {len(cache_state.perception_infos.perception_info)}).")
            try:
                box = cache_state.perception_infos.perception_info[index]["box"]  # Expecting [x1, y1, x2, y2]
                if len(box) != 4:
                    raise ValueError(f"Box data for index {position_param} has incorrect format: {box}")
                x1, y1, x2, y2 = box
                # Calculate center coordinates as floats
                cx = x1 + (x2 - x1) * rx
                cy = y1 + (y2 - y1) * ry
                return cx, cy
            except KeyError:
                raise ValueError(f"Key 'box' not found in perception_infos for index {position_param}.")
            except TypeError as e:
                raise ValueError(f"Error processing box data for index {position_param}: {e}")
        else:
            raise TypeError(f"Position must be an integer index or a tuple (float, float), got {type(position_param)}")

    def _resolve_position_with_jitter(self, position_param: PositionType, system_context: AndroidContext,
                                      jitter_px: float = 5.0) -> Tuple[float, float]:
        """
        论文 Algorithm 1 第4行: absolute 坐标模式添加有界扰动 ε (‖ε‖∞ ≤ 5px)
        在 _resolve_position_to_coords 基础上对 absolute 坐标添加随机扰动，
        用于逃脱边缘/遮挡等极端情况。
        """
        cx, cy = self._resolve_position_to_coords(position_param, system_context)
        # 仅对 absolute 坐标 (len==2的tuple/list) 添加 jitter
        if (isinstance(position_param, (tuple, list)) and len(position_param) == 2):
            eps_x = random.uniform(-jitter_px, jitter_px)
            eps_y = random.uniform(-jitter_px, jitter_px)
            # 确保不越界
            cache_state = system_context.cached_state
            if cache_state and cache_state.screenshot_dim:
                image_w, image_h = cache_state.screenshot_dim
                cx = max(0, min(image_w, cx + eps_x))
                cy = max(0, min(image_h, cy + eps_y))
            else:
                cx += eps_x
                cy += eps_y
            logger.debug(f"Applied jitter: ε=({eps_x:.1f}, {eps_y:.1f}), final=({cx:.1f}, {cy:.1f})")
        return cx, cy

    def _try_encoding(self, action_data: Dict[str, Any], encoding: str,
                      system_context: AndroidContext) -> Optional[Tuple[float, float]]:
        """
        论文 Algorithm 1 第3-5行: 根据指定编码类型解析坐标点。
        三种编码优先级: index(centroid) → relative(in-box) → absolute(screen)+jitter

        Args:
            action_data: LLM输出的动作参数字典
            encoding: 编码类型 'index' | 'relative' | 'absolute'
            system_context: 当前系统上下文

        Returns:
            解析成功返回 (cx, cy) 坐标，失败返回 None
        """
        try:
            action_name = list(action_data.keys())[0]
            params = action_data[action_name]

            # 提取 position 参数（click/long_press 为 position，swipe/drag 为 start_position）
            position = params.get('position', params.get('start_position'))
            if position is None:
                return None

            cache_state = system_context.cached_state
            if not cache_state or not cache_state.perception_infos:
                return None

            if encoding == 'index' and isinstance(position, int):
                # 优先使用 index centroid
                cx, cy = self._resolve_position_to_coords(position, system_context)
                logger.info(f"[三级降级] index 编码成功: ({cx:.1f}, {cy:.1f})")
                return (cx, cy)

            elif encoding == 'relative' and isinstance(position, int):
                # 使用 index 的 bbox 内随机采样 (relative in-box)
                index = position - 1
                perception_info = cache_state.perception_infos.perception_info
                if 0 <= index < len(perception_info):
                    box = perception_info[index].get("box")
                    if box and len(box) == 4:
                        x1, y1, x2, y2 = box
                        lambda_w = random.uniform(0.2, 0.8)  # 避免极端边缘
                        lambda_h = random.uniform(0.2, 0.8)
                        cx = x1 + (x2 - x1) * lambda_w
                        cy = y1 + (y2 - y1) * lambda_h
                        logger.info(f"[三级降级] relative 编码成功: ({cx:.1f}, {cy:.1f}), "
                                    f"λ=({lambda_w:.2f}, {lambda_h:.2f})")
                        return (cx, cy)
                return None

            elif encoding == 'absolute':
                # 使用 absolute 坐标 + jitter 扰动
                if isinstance(position, int):
                    # 将 index 转换为 absolute 坐标后加 jitter
                    cx, cy = self._resolve_position_to_coords(position, system_context)
                    eps_x = random.uniform(-5.0, 5.0)
                    eps_y = random.uniform(-5.0, 5.0)
                    if cache_state.screenshot_dim:
                        image_w, image_h = cache_state.screenshot_dim
                        cx = max(0, min(image_w, cx + eps_x))
                        cy = max(0, min(image_h, cy + eps_y))
                    logger.info(f"[三级降级] absolute+ε 编码成功: ({cx:.1f}, {cy:.1f}), "
                                f"ε=({eps_x:.1f}, {eps_y:.1f})")
                    return (cx, cy)
                elif isinstance(position, (tuple, list)) and len(position) == 2:
                    cx, cy = self._resolve_position_with_jitter(position, system_context, jitter_px=5.0)
                    logger.info(f"[三级降级] absolute+ε 编码成功: ({cx:.1f}, {cy:.1f})")
                    return (cx, cy)
                return None

            return None

        except Exception as e:
            logger.debug(f"[三级降级] {encoding} 编码失败: {e}")
            return None

    def get_fallback_encodings(self, action_data: Dict[str, Any]) -> List[str]:
        """
        论文 Algorithm 1 第1行: 构建降级候选序列 Π = [index, relative, absolute]
        根据动作数据中 position 的类型来决定候选编码顺序。
        """
        action_name = list(action_data.keys())[0]
        params = action_data[action_name]
        position = params.get('position', params.get('start_position'))

        if isinstance(position, int):
            # index → relative(in-box sampling) → absolute+ε
            return ['index', 'relative', 'absolute']
        elif isinstance(position, (tuple, list)) and len(position) == 3:
            # 已经是 relative 编码，降级到 absolute
            return ['relative', 'absolute']
        elif isinstance(position, (tuple, list)) and len(position) == 2:
            # 已经是 absolute 编码，只加 jitter 重试
            return ['absolute']
        else:
            return ['index']

    def register_custom_actions(self):
        @self.registry.action(
            description="Click on screen at a specified position.",
            param_model=ClickAction
        )
        def click(params: ClickAction, system_context: AndroidContext):
            try:
                cx, cy = self._resolve_position_to_coords(params.position, system_context)
                system_context.click(cx, cy)
                if self.highlight_action:
                    resolved_coords_dict = {'pos': (cx, cy)}
                    system_context.highlight_action('click', params, resolved_coords_dict)
                if isinstance(params.position, int):
                    return ActionResult(
                        extracted_content=f"Clicked on the center of highlight box index: {params.position}")
                else:
                    return ActionResult(extracted_content=f"Clicked on coordinates: ({cx:.2f}, {cy:.2f})")
            except (ValueError, IndexError, TypeError) as e:
                error_msg = f"Failed to resolve click position ({params.position}): {e}"
                # Log the error internally as well if needed
                # logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return ActionResult(error=error_msg, include_in_memory=True)
            except Exception as e:
                error_msg = f"An unexpected error occurred during click: {e}"
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action(
            description="Press a special hardware/navigation key ('home', 'back', or 'recent').",
            param_model=PressKeyAction
        )
        def press_key(params: PressKeyAction, system_context: AndroidContext):
            try:
                key = params.key_name
                system_context.press_key(key)
                if self.highlight_action:
                    system_context.highlight_action('press_key', params)
                return ActionResult(extracted_content=f"Pressed key: '{key}'")
            except Exception as e:
                error_msg = f"An unexpected error occurred during press_key '{params.key_name}': {e}"
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action(
            description="Input text into a specified position if the position param is provided, else input text into the currently focused element.",
            param_model=InputTextAction
        )
        def input_text(params: InputTextAction, system_context: AndroidContext):
            try:
                if params.position:
                    cx, cy = self._resolve_position_to_coords(params.position, system_context)
                    system_context.click(cx, cy)
                    resolved_coords_dict = {'pos': (cx, cy)}
                    time.sleep(0.5)
                else:
                    img_w, img_h = system_context.cached_state.perception_infos.perception_dim
                    cx = img_w // 2
                    cy = img_h // 2
                    resolved_coords_dict = {'pos': (cx, cy)}
                text_to_input = params.text
                clear_field = params.clear
                system_context.input_text(text_to_input, clear=clear_field)
                if self.highlight_action:
                    system_context.highlight_action('input_text', params, resolved_coords_dict)
                action = "Cleared field and inputted" if clear_field else "Inputted"
                # Truncate long text in message for brevity
                text_preview = (text_to_input[:50] + '...') if len(text_to_input) > 50 else text_to_input
                return ActionResult(extracted_content=f"{action} text: '{text_preview}'")
            except Exception as e:
                error_msg = f"An unexpected error occurred during input_text: {e}"
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action(
            description="Swipe on screen from a start position to an end position.",
            param_model=SwipeAction
        )
        def swipe(params: SwipeAction, system_context: AndroidContext):
            try:
                fx, fy = self._resolve_position_to_coords(params.start_position, system_context)
                tx, ty = self._resolve_position_to_coords(params.end_position, system_context)

                # Optional: Add check to prevent swipe if start and end are the same
                if abs(fx - tx) < 1 and abs(fy - ty) < 1:
                    return ActionResult(
                        error=f"Swipe cancelled: Start ({fx:.1f},{fy:.1f}) and end ({tx:.1f},{ty:.1f}) positions are too close.",
                        include_in_memory=False)

                system_context.swipe(fx, fy, tx, ty, duration=0.2)
                if self.highlight_action:
                    resolved_coords_dict = {'start': (fx, fy), 'end': (tx, ty)}
                    system_context.highlight_action('swipe', params, resolved_coords_dict)
                return ActionResult(
                    extracted_content=f"Swiped from ({fx:.1f}, {fy:.1f}) to ({tx:.1f}, {ty:.1f}). Start: '{params.start_position}', End: '{params.end_position}'")
            except (ValueError, IndexError, TypeError) as e:
                error_msg = f"Failed to resolve swipe positions (Start: {params.start_position}, End: {params.end_position}): {e}"
                return ActionResult(error=error_msg, include_in_memory=True)
            except Exception as e:
                error_msg = f"An unexpected error occurred during swipe: {e}"
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action(
            description="Drag an object on screen from a start position to an end position.",
            param_model=DragAction
        )
        def drag(params: DragAction, system_context: AndroidContext):
            try:
                fx, fy = self._resolve_position_to_coords(params.start_position, system_context)
                tx, ty = self._resolve_position_to_coords(params.end_position, system_context)

                # Optional: Add check to prevent drag if start and end are the same
                if abs(fx - tx) < 1 and abs(fy - ty) < 1:
                    return ActionResult(
                        error=f"Drag cancelled: Start ({fx:.1f},{fy:.1f}) and end ({tx:.1f},{ty:.1f}) positions are too close.",
                        include_in_memory=True)

                system_context.drag(fx, fy, tx, ty, duration=1.0)
                if self.highlight_action:
                    resolved_coords_dict = {'start': (fx, fy), 'end': (tx, ty)}
                    system_context.highlight_action('drag', params, resolved_coords_dict)
                return ActionResult(
                    extracted_content=f"Dragged from ({fx:.1f}, {fy:.1f}) to ({tx:.1f}, {ty:.1f}). Start: '{params.start_position}', End: '{params.end_position}'")
            except (ValueError, IndexError, TypeError) as e:
                error_msg = f"Failed to resolve drag positions (Start: {params.start_position}, End: {params.end_position}): {e}"
                return ActionResult(error=error_msg, include_in_memory=True)
            except Exception as e:
                error_msg = f"An unexpected error occurred during drag: {e}"
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action(
            description="Perform a long press (long click) at a specified position for a given duration.",
            param_model=LongPressAction
        )
        def long_press(params: LongPressAction, system_context: AndroidContext):
            try:
                cx, cy = self._resolve_position_to_coords(params.position, system_context)
                duration = params.duration
                system_context.long_press(cx, cy, duration=duration)
                if self.highlight_action:
                    resolved_coords_dict = {'pos': (cx, cy)}
                    system_context.highlight_action('long_press', params, resolved_coords_dict)

                if isinstance(params.position, int):
                    pos_repr = f"index {params.position}"
                else:
                    pos_repr = f"coordinates ({cx:.1f}, {cy:.1f})"

                return ActionResult(extracted_content=f"Performed long press at {pos_repr} for {duration}s.")

            except (ValueError, IndexError, TypeError) as e:
                error_msg = f"Failed to resolve long press position ({params.position}): {e}"
                return ActionResult(error=error_msg, include_in_memory=True)
            except Exception as e:
                error_msg = f"An unexpected error occurred during long press: {e}"
                # Consider logging traceback here if needed
                # logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return ActionResult(error=error_msg, include_in_memory=True)
