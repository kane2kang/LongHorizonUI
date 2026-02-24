import time
import traceback
from typing import Union, Tuple, Optional, Literal

from .base_controller import BaseController
from .views import ActionResult, ClickAction, PositionType, LongPressAction, SwipeAction, \
    InputTextAction, PressKeyAction, DragAction, OpenAppAction,ScrollAction, NavigateBackAction, WaitAction
from ..system.android_context import AndroidContext


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

        @self.registry.action(
            description="Open a specified application by package name or activity name.",
            param_model=OpenAppAction
        )
        def open_app(params: OpenAppAction, system_context: AndroidContext):
            try:
                app_identifier = params.app_name
                system_context.open_app(app_identifier)

                if self.highlight_action:
                    # 在屏幕中央显示应用图标+名称
                    img_w, img_h = system_context.cached_state.perception_infos.perception_dim
                    resolved_coords_dict = {
                        'pos': (img_w // 2, img_h // 2),
                        'app_icon': f"res/drawable/{app_identifier}.png"
                    }
                    system_context.highlight_action('open_app', params, resolved_coords_dict)

                return ActionResult(extracted_content=f"Opened application: {app_identifier}")
            except Exception as e:
                error_msg = f"Failed to open app {params.app_name}: {str(e)}"
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action(
            description="Navigate back using the system back button.",
            param_model=NavigateBackAction
        )
        def navigate_back(params: NavigateBackAction, system_context: AndroidContext):
            try:
                system_context.press_key('back')

                if self.highlight_action:
                    # 在导航栏位置高亮返回按钮
                    img_w, img_h = system_context.cached_state.perception_infos.perception_dim
                    resolved_coords_dict = {'pos': (50, img_h - 50)}
                    system_context.highlight_action('navigate_back', params, resolved_coords_dict)

                return ActionResult(extracted_content="Performed system back navigation")
            except Exception as e:
                error_msg = f"Back navigation failed: {str(e)}"
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action(
            description="Wait for a specified duration (in seconds)",
            param_model=WaitAction
        )
        def wait(params: WaitAction, system_context: AndroidContext):
            try:
                duration = params.duration
                time.sleep(duration)

                if self.highlight_action:
                    # 在全屏显示等待计时器
                    img_w, img_h = system_context.cached_state.perception_infos.perception_dim
                    resolved_coords_dict = {
                        'pos': (img_w // 2, img_h // 2),
                        'timer': f"{duration}s"
                    }
                    system_context.highlight_action('wait', params, resolved_coords_dict)

                return ActionResult(extracted_content=f"Waited for {duration} seconds")
            except Exception as e:
                error_msg = f"Wait operation interrupted: {str(e)}"
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action(
            description="Scroll on screen from start to end position",
            param_model=ScrollAction
        )
        def scroll(params: ScrollAction, system_context: AndroidContext):
            try:
                # 自动生成滚动路径（从底部80%到顶部20%）
                img_w, img_h = system_context.cached_state.perception_infos.perception_dim
                start_y = int(img_h * 0.8)
                end_y = int(img_h * 0.2)

                # 执行滚动操作（默认垂直滚动）
                system_context.swipe(img_w // 2, start_y, img_w // 2, end_y, duration=0.5)

                if self.highlight_action:
                    resolved_coords_dict = {
                        'start': (img_w // 2, start_y),
                        'end': (img_w // 2, end_y),
                        'direction': 'vertical'
                    }
                    system_context.highlight_action('scroll', params, resolved_coords_dict)

                return ActionResult(extracted_content=f"Scrolled vertically from Y={start_y} to Y={end_y}")
            except Exception as e:
                error_msg = f"Scroll operation failed: {str(e)}"
                return ActionResult(error=error_msg, include_in_memory=True)
