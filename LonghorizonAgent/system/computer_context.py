import pyautogui

pyautogui.press('shift')
import copy
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field, Field
from typing import Literal
from typing import TYPE_CHECKING, Optional, TypedDict, List, Dict
import pyperclip
import cv2
import numpy as np
import traceback

import uiautomator2 as u2
import adbutils
import platform

from .system_context import SystemState, SystemContext, SystemContextConfig


@dataclass
class ComputerContextConfig(SystemContextConfig):
    type_delay: float = 0.01
    search_wait: float = 1.0
    long_press_duration: float = 1.0
    swipe_duration: float = 0.5
    scroll_amount_multiplier: int = 100  # Adjust based on desired scroll sensitivity


logger = logging.getLogger(__name__)


class ComputerContext(SystemContext):
    def __init__(self, config: ComputerContextConfig = ComputerContextConfig()):
        super().__init__(config)

    def init_context(self):
        """Initializes context by detecting the OS and setting platform-specific keys and ratio."""
        logger.info("Initializing Computer Context...")
        self._initialized = False
        try:
            self.system = platform.system()
            if self.system == 'Darwin':  # macOS
                self.ctrl_key = "command"
                self.search_key = ["command", "space"]
                self.ratio = 2
                logger.info(
                    f"Detected macOS. CtrlKey: {self.ctrl_key}, SearchKey: {self.search_key}, Ratio: {self.ratio}")

            elif self.system == 'Windows':
                self.ctrl_key = "ctrl"
                self.search_key = ["win", "s"]  # Or just "win" on some versions
                self.ratio = 1
                logger.info(
                    f"Detected Windows. CtrlKey: {self.ctrl_key}, SearchKey: {self.search_key}, Ratio: {self.ratio}")

            elif self.system == 'Linux':
                self.ctrl_key = "ctrl"
                # Search varies greatly (Super key, Alt+F2, etc.). Defaulting to None.
                # User might need to configure this or use other methods.
                self.search_key = None
                self.ratio = 1  # Assume 1, as scaling varies widely (X11 vs Wayland, DE settings)
                logger.info(f"Detected Linux. CtrlKey: {self.ctrl_key}, SearchKey: Not Defaulted, Ratio: {self.ratio}")
                logger.warning("Linux search key not automatically set. App opening might not work reliably.")
            else:
                logger.error(f"Unsupported operating system: {self.system}")
                exit(1)

            self._initialized = True
            logger.info("Computer Context Initialized successfully.")

        except Exception as e:
            logger.error("Failed to initialize Computer Context.")
            traceback.print_exc()
            exit(1)  # Exit if critical setup fails

    def _ensure_initialized(self):
        """Internal helper to check if init_context has been called."""
        if not self._initialized:
            raise RuntimeError("Computer context not initialized. Call init_context() first.")
        if self.system is None or self.ctrl_key is None:
            raise RuntimeError("OS detection failed during initialization.")

    def _adjust_coords(self, x: float | int | None, y: float | int | None) -> tuple[int | None, int | None]:
        """Adjusts coordinates based on the detected screen ratio."""
        adj_x = int(x / self.ratio) if x is not None else None
        adj_y = int(y / self.ratio) if y is not None else None
        # Ensure coordinates are non-negative after adjustment
        adj_x = max(0, adj_x) if adj_x is not None else None
        adj_y = max(0, adj_y) if adj_y is not None else None
        return adj_x, adj_y

    def _contains_chinese(self, text: str) -> bool:
        """Checks if the input string contains any Chinese characters."""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return chinese_pattern.search(text) is not None

    def take_screenshot(self, image_path: Optional[str] = None) -> str:
        """
        Takes a screenshot of the entire screen and saves it.

        :return: The path to the saved screenshot file, or an empty string on failure.
        """
        self._ensure_initialized()
        try:
            if image_path and os.path.exists(image_path):
                return image_path
            os.makedirs(self.config.screenshot_save_dir, exist_ok=True)
            screenshot_path = os.path.join(self.config.screenshot_save_dir, f"{uuid.uuid4()}.png")
            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_path)
            logger.info(f"Screenshot saved to: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logger.error("Failed to take or save screenshot.")
            traceback.print_exc()
            return ''

    def click(self, x: float | int, y: float | int):
        """
        Performs a mouse click at the specified coordinates.
        Coordinates are absolute screen pixels.

        :param x: X-coordinate.
        :param y: Y-coordinate.
        """
        self._ensure_initialized()
        adj_x, adj_y = self._adjust_coords(x, y)
        logger.info(f"Clicking at ({x}, {y}) -> adjusted ({adj_x}, {adj_y})")
        try:
            if adj_x is None or adj_y is None:
                raise ValueError("Cannot click with None coordinates")
            pyautogui.click(adj_x, adj_y)
        except Exception as e:
            logger.error(f"Failed to click at ({x}, {y})")
            traceback.print_exc()

    def double_click(self, x: float | int, y: float | int):
        """
        Performs a double mouse click at the specified coordinates.
        Coordinates are absolute screen pixels.

        :param x: X-coordinate.
        :param y: Y-coordinate.
        """
        self._ensure_initialized()
        adj_x, adj_y = self._adjust_coords(x, y)
        logger.info(f"Double Clicking at ({x}, {y}) -> adjusted ({adj_x}, {adj_y})")
        try:
            if adj_x is None or adj_y is None:
                raise ValueError("Cannot double click with None coordinates")
            pyautogui.doubleClick(adj_x, adj_y)
        except Exception as e:
            logger.error(f"Failed to double click at ({x}, {y})")
            traceback.print_exc()

    def right_click(self, x: float | int, y: float | int):
        """
        Performs a right mouse click at the specified coordinates.
        Coordinates are absolute screen pixels.

        :param x: X-coordinate.
        :param y: Y-coordinate.
        """
        self._ensure_initialized()
        adj_x, adj_y = self._adjust_coords(x, y)
        logger.info(f"Right Clicking at ({x}, {y}) -> adjusted ({adj_x}, {adj_y})")
        try:
            if adj_x is None or adj_y is None:
                raise ValueError("Cannot right click with None coordinates")
            pyautogui.rightClick(adj_x, adj_y)
        except Exception as e:
            logger.error(f"Failed to right click at ({x}, {y})")
            traceback.print_exc()

    def long_press(self, x: float | int, y: float | int, duration: float | None = None):
        """
        Performs a long press (mouse down, wait, mouse up) at specified coordinates.
        Coordinates are absolute screen pixels.

        :param x: X-coordinate.
        :param y: Y-coordinate.
        :param duration: Duration to hold the mouse down in seconds. Uses config default if None.
        """
        self._ensure_initialized()
        press_duration = duration if duration is not None else self.config.long_press_duration
        adj_x, adj_y = self._adjust_coords(x, y)
        logger.info(f"Long pressing at ({x}, {y}) -> adjusted ({adj_x}, {adj_y}) for {press_duration}s")
        try:
            if adj_x is None or adj_y is None:
                raise ValueError("Cannot long press with None coordinates")
            pyautogui.moveTo(adj_x, adj_y)
            pyautogui.mouseDown(button='left')
            time.sleep(press_duration)
            pyautogui.mouseUp(button='left')
        except Exception as e:
            logger.error(f"Failed to long press at ({x}, {y})")
            traceback.print_exc()

    def swipe(self, fx: float | int, fy: float | int, tx: float | int, ty: float | int, duration: float | None = None):
        """
        Swipes (drags) the mouse from a start point to an end point.
        Coordinates are absolute screen pixels.

        :param fx: Starting X-coordinate.
        :param fy: Starting Y-coordinate.
        :param tx: Ending X-coordinate.
        :param ty: Ending Y-coordinate.
        :param duration: Duration of the swipe in seconds. Uses config default if None.
        """
        self._ensure_initialized()
        swipe_duration = duration if duration is not None else self.config.swipe_duration
        adj_fx, adj_fy = self._adjust_coords(fx, fy)
        adj_tx, adj_ty = self._adjust_coords(tx, ty)
        logger.info(
            f"Swiping from ({fx}, {fy}) -> ({adj_fx}, {adj_fy}) to ({tx}, {ty}) -> ({adj_tx}, {adj_ty}) over {swipe_duration}s")
        try:
            if None in [adj_fx, adj_fy, adj_tx, adj_ty]:
                raise ValueError("Cannot swipe with None coordinates")
            pyautogui.moveTo(adj_fx, adj_fy)  # Move to start position first
            pyautogui.dragTo(adj_tx, adj_ty, duration=swipe_duration, button='left')
        except Exception as e:
            logger.error(f"Failed to swipe from ({fx}, {fy}) to ({tx}, {ty})")
            traceback.print_exc()

    def input_text(self, text: str, delay: float | None = None):
        """
        Types the given text. Handles Chinese characters via clipboard paste.

        :param text: The text to type.
        :param delay: Delay between keystrokes for non-Chinese text. Uses config default if None.
        """
        self._ensure_initialized()
        typing_delay = delay if delay is not None else self.config.type_delay
        logger.info(f"Typing text (length: {len(text)})")
        try:
            if self._contains_chinese(text):
                logger.debug("Detected Chinese text, using clipboard paste.")
                pyperclip.copy(text)
                time.sleep(0.1)  # Small delay for clipboard
                pyautogui.hotkey(self.ctrl_key, 'v')  # Use hotkey for simplicity
                time.sleep(0.1)  # Small delay after paste
            else:
                logger.debug(f"Typing using pyautogui.typewrite with interval {typing_delay}s.")
                pyautogui.typewrite(text, interval=typing_delay)
        except Exception as e:
            logger.error(f"Failed to type text: {text[:50]}...")  # Log truncated text
            traceback.print_exc()

    def press_key(self, key: str | list[str]):
        """
        Presses a single key or a sequence of keys.

        :param key: Key name (e.g., 'enter', 'backspace', 'win') or list of keys.
        """
        self._ensure_initialized()
        logger.info(f"Pressing key(s): {key}")
        try:
            if isinstance(key, list):
                pyautogui.press(key)
            else:
                pyautogui.press(key)
        except Exception as e:
            logger.error(f"Failed to press key(s): {key}")
            traceback.print_exc()

    def key_down(self, key: str):
        """Holds down a key."""
        self._ensure_initialized()
        logger.debug(f"Holding key down: {key}")
        try:
            pyautogui.keyDown(key)
        except Exception as e:
            logger.error(f"Failed to hold key down: {key}")
            traceback.print_exc()

    def key_up(self, key: str):
        """Releases a key."""
        self._ensure_initialized()
        logger.debug(f"Releasing key: {key}")
        try:
            pyautogui.keyUp(key)
        except Exception as e:
            logger.error(f"Failed to release key: {key}")
            traceback.print_exc()

    def shortcut(self, *keys):
        """
        Performs a keyboard shortcut (presses keys simultaneously).

        :param keys: Keys involved in the shortcut (e.g., 'ctrl', 'c').
        """
        self._ensure_initialized()
        logger.info(f"Performing shortcut: {'+'.join(keys)}")
        try:
            pyautogui.hotkey(*keys)
        except Exception as e:
            logger.error(f"Failed to perform shortcut: {'+'.join(keys)}")
            traceback.print_exc()

    def scroll(self, amount: int, x: float | int | None = None, y: float | int | None = None):
        """
        Scrolls the mouse wheel vertically. Positive amount scrolls up, negative scrolls down.
        Optionally moves the mouse cursor first.

        :param amount: Number of scroll "clicks". Positive for up, negative for down.
                       The actual distance scrolled depends on the OS/app settings.
        :param x: Optional X-coordinate to move mouse before scrolling.
        :param y: Optional Y-coordinate to move mouse before scrolling.
        """
        self._ensure_initialized()
        direction = "up" if amount > 0 else "down"
        log_msg = f"Scrolling {direction} by {abs(amount)} clicks"
        adj_x, adj_y = None, None
        if x is not None and y is not None:
            adj_x, adj_y = self._adjust_coords(x, y)
            log_msg += f" at ({x}, {y}) -> adjusted ({adj_x}, {adj_y})"
        logger.info(log_msg)
        try:
            if adj_x is not None and adj_y is not None:
                pyautogui.moveTo(adj_x, adj_y)  # Move cursor if coords are given
            # PyAutoGUI scroll: positive up, negative down. Amount is 'clicks'.
            # Multiply by config factor for potentially larger scrolls per call
            scroll_value = amount * self.config.scroll_amount_multiplier
            pyautogui.scroll(scroll_value)
        except Exception as e:
            logger.error(f"Failed to scroll.")
            traceback.print_exc()

    def move_to(self, x: float | int, y: float | int, duration: float = 0.1):
        """
        Moves the mouse cursor to the specified coordinates instantly or over a duration.
        Coordinates are absolute screen pixels.

        :param x: X-coordinate.
        :param y: Y-coordinate.
        :param duration: Time in seconds to take moving the mouse. 0 for instant.
        """
        self._ensure_initialized()
        adj_x, adj_y = self._adjust_coords(x, y)
        logger.debug(f"Moving mouse to ({x}, {y}) -> adjusted ({adj_x}, {adj_y}) over {duration}s")
        try:
            if adj_x is None or adj_y is None:
                raise ValueError("Cannot move mouse to None coordinates")
            pyautogui.moveTo(adj_x, adj_y, duration=duration)
        except Exception as e:
            logger.error(f"Failed to move mouse to ({x}, {y})")
            traceback.print_exc()

    def get_mouse_position(self) -> tuple[int, int]:
        """
        Gets the current mouse cursor position.

        :return: Tuple (x, y) of the current mouse coordinates, adjusted back by the ratio.
        """
        self._ensure_initialized()
        try:
            current_x_adj, current_y_adj = pyautogui.position()
            # Apply ratio back to get screen coordinates expected by user
            final_x = int(current_x_adj * self.ratio)
            final_y = int(current_y_adj * self.ratio)
            logger.debug(
                f"Current mouse position (adjusted): ({current_x_adj}, {current_y_adj}) -> reported: ({final_x}, {final_y})")
            return final_x, final_y
        except Exception as e:
            logger.error("Failed to get mouse position.")
            traceback.print_exc()
            return (0, 0)  # Return default on error

    def open_app(self, app_name: str):
        """
        Attempts to open an application using the system's search function.
        Reliability depends on the OS search behavior and `search_key` setting.

        :param app_name: The name of the application to search for and open.
        """
        self._ensure_initialized()
        if not self.search_key:
            logger.error(f"Cannot open app '{app_name}': Search key not defined for OS '{self.system}'.")
            return

        logger.info(f"Attempting to open app: {app_name} using keys: {self.search_key}")
        try:
            # Press search shortcut
            pyautogui.hotkey(*self.search_key)
            time.sleep(0.5)  # Wait for search bar to appear

            # Type application name
            self.input_text(app_name, delay=0.05)  # Use internal type method for consistency
            time.sleep(self.config.search_wait)  # Wait for search results

            # Press Enter
            self.press_key('enter')
            logger.info(f"Sent 'Enter' key to hopefully launch {app_name}.")
            time.sleep(1.0)  # Give app time to start opening

        except Exception as e:
            logger.error(f"Failed to open app '{app_name}'.")
            traceback.print_exc()
