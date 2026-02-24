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

import cv2
import numpy as np
import traceback
from typing import Tuple, Dict, Optional

import uiautomator2 as u2
import adbutils
from threading import Lock
from pathlib import Path
from .system_context import SystemState, SystemContext, SystemContextConfig


@dataclass
class AndroidContextConfig(SystemContextConfig):
    device_id: str | adbutils.AdbDevice | None = None
    local_screenshot_dir: Optional[str] = None  # 本地截图目录路径


logger = logging.getLogger(__name__)


class AndroidContext(SystemContext):
    def __init__(self, config: AndroidContextConfig):
        super().__init__(config)
        self.config = config
        self.screenshot_files = []
        self.current_screenshot_index = 0
        self.screenshot_files = []
        self.current_screenshot_index = 0
        self._load_local_screenshots()  # 预加载截图

        if self.config.local_screenshot_dir:
            screenshot_dir = Path(self.config.local_screenshot_dir)
            self.screenshot_files = sorted(
                [str(p) for p in screenshot_dir.glob("*.png")],
                key=lambda x: os.path.basename(x)
            )
            logger.info(f"Loaded {len(self.screenshot_files)} local screenshots from {screenshot_dir}")
        self._current_step = 0

    def _load_local_screenshots(self):
        """加载本地截图目录中的所有PNG文件"""
        if not self.config.local_screenshot_dir:
            self.screenshot_files = []
            logger.warning("No local screenshot directory configured")
            return

        try:
            screenshot_dir = Path(self.config.local_screenshot_dir)
            if not screenshot_dir.exists():
                logger.error(f"Local screenshot directory does not exist: {screenshot_dir}")
                self.screenshot_files = []
                return
            self.screenshot_files = sorted(
                [str(p) for p in screenshot_dir.glob("*.png")],
                key=lambda x: os.path.basename(x)  # 按文件名排序
            )
            logger.info(f"Loaded {len(self.screenshot_files)} screenshots from {screenshot_dir}")
        except Exception as e:
            logger.error(f"Failed to load local screenshots: {e}")
            self.screenshot_files = []

    def init_context(self):
        self._context = None
        logger.info("Running in local screenshot mode")

    def increment_step(self):
        """步骤计数器递增"""
        if not hasattr(self, '_current_step'):
            self._current_step = 0  # 初始化
        self._current_step += 1

    def get_step_prefix(self) -> str:
        """生成三位步数前缀"""
        return f"{self._current_step:03}_"

    def take_screenshot(self, image_path: Optional[str] = None) -> str:
        """从本地文件加载截图"""
        self.increment_step()

        if image_path and os.path.exists(image_path):
            return image_path

        if self.screenshot_files and self.current_screenshot_index < len(self.screenshot_files):
            screenshot_path = self.screenshot_files[self.current_screenshot_index]
            print(f"Using local screenshot [{self.current_screenshot_index}]: {screenshot_path}")
            self.current_screenshot_index += 1
            return screenshot_path

        try:
            if self.config.screenshot_save_dir:
                print()
                os.makedirs(self.config.screenshot_save_dir, exist_ok=True)
                prefix = self.get_step_prefix()
                filename = f"{prefix}placeholder.png"
                screenshot_path = os.path.join(self.config.screenshot_save_dir, filename)
                cv2.imwrite(screenshot_path, np.zeros((100, 100, 3), dtype=np.uint8))
                print(f"Created placeholder screenshot: {screenshot_path}")
                return screenshot_path
        except Exception as e:
            print(f"Failed to create placeholder: {e}")

        return ""


    def _ensure_connected(self):
        """Internal helper to check for active connection."""
        # if not self._context :
        #     raise ConnectionError("Device not connected. Initialize context first.")

    def install_app(self, app_path: str):
        """
        Installs an application (.apk) onto the device.
        Logs a warning if the path is invalid or not an .apk file.
        :param app_path: Path to the .apk file (local path or URL).
        """
        self._ensure_connected()
        # Basic check for local files, URLs might bypass this
        if not os.path.exists(app_path) or not app_path.lower().endswith(".apk"):
            logger.warning(f"Invalid local app path or file type: {app_path}")

        logger.info(f"Attempting to install app from: {app_path}")
        try:
            self._context.app_install(app_path)
            logger.info(f"Successfully initiated installation of app: {app_path}")
        except Exception as e:
            logger.error(f"Failed to install app {app_path}: {e}")
            traceback.print_exc()

    def click(self, x: float, y: float):
        """
        Performs a click at the specified coordinates.
        Coordinates can be absolute pixels or relative (0.0 to 1.0).
        :param x: X-coordinate.
        :param y: Y-coordinate.
        """
        if self._context is None:
            return
        self._ensure_connected()
        logger.info(f"Clicking at x: {x}, y: {y}")
        try:
            self._context.click(x, y)
        except Exception as e:
            logger.error(f"Failed to click at ({x}, {y}): {e}")
            traceback.print_exc()

    def long_press(self, x: float, y: float, duration: float = 1.0):
        """
        Performs a long press (long click) at the specified coordinates.
        :param x: X-coordinate.
        :param y: Y-coordinate.
        :param duration: Duration of the long press in seconds (default: 1.0).
        """
        if self._context is None:
            logger.error("Context is not available. Cannot perform long press.")
            return
        self._ensure_connected()
        logger.info(f"Long pressing at x: {x}, y: {y}, duration: {duration}s")
        try:
            # uiautomator2 method name is long_click
            self._context.long_click(x, y, duration=duration)
        except Exception as e:
            logger.error(f"Failed to long press at ({x}, {y}): {e}")
            traceback.print_exc()

    def swipe(self, fx: float, fy: float, tx: float, ty: float, duration: float = 0.5):
        """
        Swipes from a start point (fx, fy) to an end point (tx, ty).
        Coordinates can be absolute pixels or relative (0.0 to 1.0).
        :param fx: Starting X-coordinate.
        :param fy: Starting Y-coordinate.
        :param tx: Ending X-coordinate.
        :param ty: Ending Y-coordinate.
        :param duration: Duration of the swipe in seconds (default: 0.5).
        """
        if self._context is None:
            logger.error("Context is not available. Cannot perform swipe.")
            return
        self._ensure_connected()
        logger.info(f"Swiping from ({fx}, {fy}) to ({tx}, {ty}) with duration {duration}s")
        try:
            self._context.swipe(fx, fy, tx, ty, duration=duration)
        except Exception as e:
            logger.error(f"Failed to swipe: {e}")
            traceback.print_exc()

    def drag(self, fx: float, fy: float, tx: float, ty: float, duration: float = 2.):
        """
        Drags from a start point (fx, fy) to an end point (tx, ty).
        Similar to swipe but might imply interaction with draggable elements.
        Coordinates can be absolute pixels or relative (0.0 to 1.0).
        :param fx: Starting X-coordinate.
        :param fy: Starting Y-coordinate.
        :param tx: Ending X-coordinate.
        :param ty: Ending Y-coordinate.
        :param duration: Duration of the drag in seconds (default: 0.5).
        """
        if self._context is None:
            logger.error("Context is not available. Cannot perform drag.")
            return
        self._ensure_connected()
        logger.info(f"Dragging from ({fx}, {fy}) to ({tx}, {ty}) with duration {duration}s")
        try:
            self._context.drag(fx, fy, tx, ty, duration=duration)
        except Exception as e:
            logger.error(f"Failed to drag: {e}")
            traceback.print_exc()

    def input_text(self, text: str, clear: bool = False):
        """
        Inputs text, likely into the currently focused input field.
        Use click or element selection first to focus the correct field if needed.
        :param text: The text string to input.
        :param clear: If True, clears the existing text before inputting (default: False).
        """
        if self._context is None:
            logger.error("Context is not available. Cannot input text.")
            return
        self._ensure_connected()
        logger.info(f"Inputting text: '{text}' (Clear existing: {clear})")
        try:
            # send_keys simulates keyboard input
            self._context.send_keys(text, clear=clear)
        except Exception as e:
            logger.error(f"Failed to input text '{text}': {e}")
            traceback.print_exc()

    def press_key(self, key_code: str):
        """
        Simulates pressing a device key (hardware or software).
        Common key_codes: 'home', 'back', 'enter', 'delete', 'volume_up', 'volume_down', 'power'.
        Can also be an Android KeyEvent code (integer).
        :param key_code: The key name (string) or key code (integer).
        """
        if self._context is None:
            logger.error("Context is not available. Cannot input text.")
            return
        self._ensure_connected()
        logger.info(f"Pressing key: {key_code}")
        try:
            self._context.press(key_code)
        except Exception as e:
            logger.error(f"Failed to press key '{key_code}': {e}")
            traceback.print_exc()

    def push_file(self, local_path: str, device_path: str, mode: int = 0o755):
        """
        Pushes a file or directory from the local machine to the device.
        :param local_path: Path to the local file or directory.
        :param device_path: Destination path on the device.
        :param mode: File mode for the pushed file on the device (default: 0o755).
        :return: True if successful, False otherwise.
        """
        self._ensure_connected()
        if not os.path.exists(local_path):
            logger.error(f"Local path does not exist: {local_path}")
            return False
        logger.info(f"Pushing '{local_path}' to device '{device_path}' with mode {oct(mode)}")
        try:
            # uiautomator2's push handles the underlying adb push
            self._context.push(local_path, device_path, mode=mode)
            logger.info(f"Successfully pushed file to {device_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to push file from {local_path} to {device_path}: {e}")
            traceback.print_exc()
            return False

    def pull_file(self, device_path: str, local_path: str) -> bool:
        """
        Pulls a file or directory from the device to the local machine.
        :param device_path: Path to the file or directory on the device.
        :param local_path: Destination path on the local machine. The directory structure will be created if it doesn't exist.
        :return: True if successful, False otherwise.
        """
        self._ensure_connected()
        # Ensure local directory exists
        local_dir = os.path.dirname(local_path)
        if local_dir and not os.path.exists(local_dir):
            try:
                os.makedirs(local_dir, exist_ok=True)
                logger.info(f"Created local directory: {local_dir}")
            except OSError as e:
                logger.error(f"Failed to create local directory {local_dir}: {e}")
                return False

        logger.info(f"Pulling '{device_path}' from device to '{local_path}'")
        try:
            # uiautomator2's pull handles the underlying adb pull
            self._context.pull(device_path, local_path)
            logger.info(f"Successfully pulled file to {local_path}")
            return True
        except Exception as e:
            # Specific check if remote path doesn't exist might require a shell command first
            logger.error(f"Failed to pull file from {device_path} to {local_path}: {e}")
            traceback.print_exc()
            return False

    def execute_shell_command(self, command: str, timeout: Optional[float] = 60.0) -> Optional[Tuple[str, int]]:
        """
        Executes an ADB shell command on the device.
        :param command: The shell command string to execute.
        :param timeout: Maximum execution time in seconds (default: 60.0). Use None for no timeout.
        :return: A tuple containing (stdout_output, exit_code), or None if the command failed to execute.
        """
        self._ensure_connected()
        logger.info(f"Executing shell command: '{command}' with timeout {timeout}s")
        try:
            # The shell method returns a Response object with output and exit_code attributes
            response = self._context.shell(command, timeout=timeout)
            output = response.output
            exit_code = response.exit_code
            log_level = logging.INFO if exit_code == 0 else logging.WARNING
            logger.log(log_level, f"Shell command finished. Exit code: {exit_code}")
            # Optionally log output, truncate if too long
            output_preview = (output[:100] + '...') if len(output) > 100 else output
            logger.debug(f"Shell output (preview): {output_preview.strip()}")
            return output, exit_code
        except Exception as e:
            logger.error(f"Failed to execute shell command '{command}': {e}")
            traceback.print_exc()
            return None

    def current_app(self) -> Optional[Dict]:
        """
        Gets information about the application currently in the foreground.
        :return: A dictionary containing info like 'package', 'activity', 'pid', or None on failure.
        """
        self._ensure_connected()
        logger.info("Getting current foreground app info")
        try:
            app_info = self._context.app_current()
            logger.info(f"Current app info: {app_info}")
            return app_info
        except Exception as e:
            logger.error(f"Failed to get current app info: {e}")
            traceback.print_exc()
            return None

    def get_device_info(self) -> Optional[Dict]:
        """
        Retrieves the detailed device information dictionary.
        Contains info like model, brand, resolution, sdk version, etc.
        :return: A dictionary containing device info, or None on failure.
        """
        self._ensure_connected()
        logger.info("Getting device information")
        try:
            info = self._context.device_info
            if not info:
                logger.warning("Device info dictionary is empty or None.")
                return None
            logger.info(f"Device info retrieved successfully (Serial: {info.get('serial')})")
            return info
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            traceback.print_exc()
            return None

    def swipe_ext(self, direction: Literal["left", "right", "up", "down"],
                  scale: float = 0.8,
                  box: Optional[Tuple[int, int, int, int]] = None,
                  duration: float = 0.2):
        """
        Performs a swipe in a specified direction ('left', 'right', 'up', 'down').

        Allows specifying a bounding box and a scale factor for the swipe distance.
        The scale determines the swipe length relative to the screen/box dimension.
        For 'up'/'down', starts near the vertical midpoint as suggested for stability.

        :param direction: Swipe direction: "left", "right", "up", or "down".
        :param scale: Proportional length of the swipe relative to screen/box dimension (0.0 < scale <= 1.0, default: 0.9).
        :param box: Optional tuple (x1, y1, x2, y2) defining the rectangular area to swipe within.
                    If None, swipes relative to the full screen (default: None).
        :param duration: Duration of the swipe animation in seconds (default: 0.2).
        """
        self._ensure_connected()
        logger.info(
            f"Performing extended swipe: direction='{direction}', scale={scale}, box={box}, duration={duration}s")

        if not (0 < scale <= 1.0):
            logger.error(f"Invalid scale value: {scale}. Must be between 0 (exclusive) and 1.0 (inclusive).")
            raise ValueError("Scale must be between 0 (exclusive) and 1.0 (inclusive).")

        valid_directions = ["left", "right", "up", "down"]
        if direction not in valid_directions:
            logger.error(f"Invalid direction: '{direction}'. Must be one of {valid_directions}.")
            raise ValueError(f"Direction must be one of {valid_directions}")

        try:
            self._context.swipe_ext(direction=direction, scale=scale, box=box, duration=duration)
        except Exception as e:
            logger.error(f"Failed to swipe: {e}")
            traceback.print_exc()