import pdb

import cv2
import os
import json
import numpy as np
import logging
import copy
import math
from LonghorizonAgent.perception.screen_perception import ScreenPerception
from ..common.llm_provider import LLMProvider
from ..common import llm_provider
from ..common import utils

logger = logging.getLogger(__name__)


class RecordToComplexStepsAgent:
    """
    将操作记录转为步骤
    """

    def __init__(self, llm: LLMProvider, **kwargs):
        self.llm = llm
        self.system_prompt = kwargs.get("system_prompt",
                                        "You are a professional QA engineer analyzing user actions in a game or application. "
                                        "You need to precisely describe the user's steps for test case creation. Avoid referring to 'the user'.")
        self.screen_p = ScreenPerception(**kwargs)

    def get_user_prompt(self, action_info, task="", language="Chinese"):
        prompt = f"""
        Based on the before and after images, and the provided action information, describe the user's action in detail.

        Action information (if available, a red bounding box in the "before" image represents user click/press): 
        {action_info}

        Important:
        - The red bounding box or red dot or red arrow (if present) in the 'before' image is provided only to help you identify the target of the action. Do NOT explicitly mention them in the 'action' description.
        - If no bounding box is found, a red dot(if present) is drawn at the action point when the action is Tap, Press or Drag.
        - For swipe actions, a red arrow is drawn from the start to the end of the swipe.
        - For drag actions, a red arrow is drawn from the start to the end of the drag, and a box is drawn at the start if matched.
        - If text exists but appears multiple times on the screen, provide additional positioning information (e.g., Tapped the "参加" button located on the "幻影马戏团:第2幕" card instead of just tapped on "参加" button).
        - If no text is available, describe the location of the action (relative), and describe the shape or appearance of the clicked area or icon.
        - If the user clicks the system-specific home, back, or recent button, reply with the description of `press home/back/recent key` without describing the corresponding button.
        
        Focus on:
        - Describing the action performed in natural language. Be concise and clear.
        - Identifying the location of the action:
            - If the action involves a visible bounding box in the "before" image (provided by the 'action' data), 
              describe what icon, text, or area the bounding box is located on. No absolute positions needed.
            - If no bounding box is provided, analyze the changes between the two images to determine the action 
              and its location and the shape or appearance of the clicked area or icon. Be as specific as possible about the affected region (e.g., "tapped the 'Settings' icon in the top right corner").
        - If there is relevant text in the area of action, please include it in your description.

        When generating the `icon_caption`, consider the following:
        - Focus on the icon or element within the red bounding box or red dot or a red arrow, if present.
        - **Crucially, analyze the changes between the 'before' and 'after' images to understand the *effect* of the action.**
        - **The `icon_caption` should align with and support the `action` description.** For example, if the action is "Toggled the 'Wi-Fi' switch", the `icon_caption` should describe the switch's function (e.g., "Wi-Fi control").

        {f"Given the overall task: '{task}', explain the reasoning behind this action. How does it contribute to achieving the task?" if task else ""}

        Respond in JSON format:
        {{
            "action": "A concise, natural language description of the action performed.",
            "action_result": "A detailed description of the result of the action, such as navigation to a different page (with detailed information about this page) or triggering a function or popup. Reference the second image for more context.",
            "icon_caption": "A brief (1-2 words or a short phrase) description of the icon's purpose/function within the red bounding box, if present.  This description should also be informed by the before/after image changes and should align with the 'action' description. Otherwise, leave this blank (\"\").",
            "think": "Your reasoning for performing this action based on the task (only present if task is provided)."
        }}
        Please output in {language}.
        """
        return prompt

    def filter_action(self, action_data):
        if "action" in action_data and (
                action_data["action"].startswith("Swipe") or action_data["action"].startswith("Drag")):
            # 提取坐标
            action_str = action_data["action"]
            try:
                # 使用字符串的split和strip方法提取坐标
                coords = action_str[action_str.index('(') + 1:action_str.index(')')].split(',')
                x1, y1 = int(coords[0].strip()), int(coords[1].strip())
                coords = action_str[action_str.rindex('(') + 1:action_str.rindex(')')].split(',')
                x2, y2 = int(coords[0].strip()), int(coords[1].strip())
            except (ValueError, IndexError):
                print("Error parsing action coordinates")
                return action_data

            # 计算距离
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 如果距离小于或等于20，计算中点并修改动作为 Tap
            if distance <= 20:
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                action_data["action"] = f"Tap ({mid_x}, {mid_y})"

        return action_data

    def execute(self, record_dir, task="", **kwargs):
        execute_infos = []

        output_dir = os.path.join(record_dir, "screenshot_draw")
        os.makedirs(output_dir, exist_ok=True)
        icon_dir = os.path.join(record_dir, "icons")
        os.makedirs(icon_dir, exist_ok=True)
        perception_dir = os.path.join(record_dir, "perceptions")
        os.makedirs(perception_dir, exist_ok=True)

        action_json_file = os.path.join(record_dir, "actions.json")
        with open(action_json_file, "r", encoding="utf-8") as fin:
            action_infos = json.load(fin)

        for i, action_data in enumerate(action_infos):
            try:
                screenshot_path = os.path.normpath(action_data["screenshot"])
                screenshot_path = os.path.normpath(screenshot_path).replace('\\', '/')
                image_path = os.path.join(record_dir, "screenshot", os.path.basename(screenshot_path))
                action_data["screenshot"] = os.path.relpath(image_path, record_dir)
                image_index = int(os.path.splitext(os.path.basename(image_path))[0])
                image = cv2.imread(image_path)
                img_h, img_w = image.shape[:2]
                action_data["text"] = ""
                action_data["box"] = None
                action_data["screenshot_width"] = img_w
                action_data["screenshot_height"] = img_h

                try:
                    screen_pinfo_path = os.path.join(perception_dir, f"{image_index:03d}.json")
                    if os.path.exists(screen_pinfo_path):
                        with open(screen_pinfo_path, "r") as fin:
                            screen_perception_info = json.load(fin)
                    else:
                        screen_status = self.screen_p.run_perception(image)
                        screen_perception_info = screen_status.perception_info
                        with open(screen_pinfo_path, "w") as fw:
                            json.dump(screen_status.perception_info, fw, indent=2)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    exit(1)

                action_data = self.filter_action(action_data)

                image_draw = image.copy()

                if "action" in action_data:
                    if action_data["action"].startswith("Tap") or action_data["action"].startswith("Press"):
                        try:
                            tap_coords_str = action_data["action"].split("(")[1].split(")")[0]
                            tap_x, tap_y = map(int, tap_coords_str.split(", "))
                            tap_coords = (tap_x, tap_y)
                        except (IndexError, ValueError) as e:
                            import traceback
                            traceback.print_exc()
                            exit(1)

                        closest_box = None
                        min_distance = min(img_w, img_h) // 8

                        for pinfo_ in screen_perception_info:
                            if not pinfo_['text']:
                                continue
                            x1, y1, x2, y2 = pinfo_["box"]
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            distance = np.sqrt((center_x - tap_coords[0]) ** 2 + (center_y - tap_coords[1]) ** 2)

                            if center_x > x1 and center_x < x2 and center_y > y1 and center_y < y2 and distance < min_distance:
                                min_distance = distance
                                closest_box = pinfo_["box"]
                                action_data["text"] = pinfo_['text']
                                action_data["box"] = closest_box

                        if closest_box is None:
                            # Draw a red dot at the tap location
                            cv2.circle(image_draw, (tap_x, tap_y), 10, (0, 0, 255), -1)
                        else:
                            x1, y1, x2, y2 = closest_box
                            cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            action_data["box"] = closest_box

                    elif action_data["action"].startswith("Swipe"):
                        try:
                            coords_str = action_data["action"].split("(")[1].split(")")[0]
                            x1, y1 = map(int, coords_str.split(", "))
                            x1 = max(min(img_w, x1), 0)
                            y1 = max(min(img_h, y1), 0)
                            coords_str = action_data["action"].split("(")[2].split(")")[0]
                            x2, y2 = map(int, coords_str.split(", "))
                            x2 = max(min(img_w, x2), 0)
                            y2 = max(min(img_h, y2), 0)
                            # Draw an arrow from (x1, y1) to (x2, y2)
                            cv2.arrowedLine(image_draw, (x1, y1), (x2, y2), (0, 0, 255), 10)
                        except (IndexError, ValueError) as e:
                            import traceback
                            traceback.print_exc()
                            exit(1)

                    elif action_data["action"].startswith("Drag"):
                        try:
                            coords_str = action_data["action"].split("(")[1].split(")")[0]
                            x1, y1 = map(int, coords_str.split(", "))
                            x1 = max(min(img_w, x1), 0)
                            y1 = max(min(img_h, y1), 0)
                            coords_str = action_data["action"].split("(")[2].split(")")[0]
                            x2, y2 = map(int, coords_str.split(", "))
                            x2 = max(min(img_w, x2), 0)
                            y2 = max(min(img_h, y2), 0)
                            # Draw an arrow from (x1, y1) to (x2, y2)
                            cv2.arrowedLine(image_draw, (x1, y1), (x2, y2), (0, 0, 255), 10)

                            closest_box = None
                            min_distance = min(img_w, img_h) // 8

                            for pinfo_ in screen_perception_info:
                                if not pinfo_['text']:
                                    continue
                                bx1, by1, bx2, by2 = pinfo_["box"]
                                center_x = (bx1 + bx2) // 2
                                center_y = (by1 + by2) // 2
                                distance = np.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

                                if center_x > bx1 and center_x < bx2 and center_y > by1 and center_y < by2 and distance < min_distance:
                                    min_distance = distance
                                    closest_box = pinfo_["box"]
                                    action_data["text"] = pinfo_['text']
                                    action_data["box"] = closest_box

                            if closest_box:
                                bx1, by1, bx2, by2 = closest_box
                                cv2.rectangle(image_draw, (bx1, by1), (bx2, by2), (0, 0, 255), 5)
                        except (IndexError, ValueError) as e:
                            import traceback
                            traceback.print_exc()
                            exit(1)

                output_image_path = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_image_path, image_draw)

                logger.info(f"Processed and saved: {output_image_path}")

                next_image_path = os.path.join(record_dir, "screenshot", f"{image_index + 1:03d}.png")
                action_info = copy.deepcopy(action_data)
                del action_info["screenshot"]
                action_info = json.dumps(action_info, indent=2)
                user_prompt = self.get_user_prompt(action_info, task)
                chat_messages = self.llm.add_message("system", self.system_prompt, [])
                cur_image = utils.encode_image(output_image_path)
                nex_image = utils.encode_image(next_image_path)
                chat_messages = self.llm.add_message("user", user_prompt, chat_messages,
                                                     [cur_image, nex_image])
                output_content = self.llm.invoke(chat_messages=chat_messages,
                                                 temperature=kwargs.get("temperature", 0.5))
                action_text = output_content.replace("```json", "").replace("```", "")

                exec_info = json.loads(action_text)
                if not task:
                    exec_info["think"] = ""
                exec_info["task"] = task
                exec_info["raw_action"] = action_data
                if not action_data["text"] and exec_info['icon_caption']:
                    if action_data["box"]:
                        x1, y1, x2, y2 = action_data["box"]
                        icon_img = image[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(icon_dir, f"{exec_info['icon_caption']}.png"), icon_img)
                    elif action_data["action"].startswith("Tap") or action_data["action"].startswith("Press"):
                        tap_coords_str = action_data["action"].split("(")[1].split(")")[0]
                        tap_x, tap_y = map(int, tap_coords_str.split(", "))
                        icon_w = min(img_w, img_h) // 16
                        x1 = max(0, min(img_w, tap_x - icon_w))
                        x2 = max(0, min(img_w, tap_x + icon_w))
                        y1 = max(0, min(img_h, tap_y - icon_w))
                        y2 = max(0, min(img_h, tap_y + icon_w))
                        icon_img = image[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(icon_dir, f"{exec_info['icon_caption']}.png"), icon_img)

                execute_infos.append(exec_info)
            except Exception as e:
                logger.error(f"An unexpected error occurred processing {e}")
                break

        save_json_path = os.path.join(record_dir, "execute_infos.json")
        with open(save_json_path, "w", encoding="utf-8") as fw:
            json.dump(execute_infos, indent=2, fp=fw)
        logger.info(f"save step info to {save_json_path}")
        task_steps = ""
        task_steps_with_results = ""
        for i, exe_info in enumerate(execute_infos):
            task_steps += f"{i + 1}.{exe_info['action']}\n"
            task_steps_with_results += f"{i + 1}.{exe_info['action']} {exe_info['action_result']}\n"

        if not task:
            chat_messages = self.llm.add_message("system",
                                                 "You are a professional QA engineer analyzing user actions in a game or application. Please use a few sentences to briefly describe the task based on the steps provided by the user. Avoid repeating the specific steps and instead give a summary. Please output in Chinese",
                                                 [])
            chat_messages = self.llm.add_message("user", f"Task Steps:\n{task_steps}", chat_messages)
            task_name = self.llm.invoke(chat_messages=chat_messages, temperature=kwargs.get("temperature", 0.5))

            logger.info(f"task name:\n{task_name}")
        logger.info(f"task steps:\n{task_steps}")
        logger.info(f"task steps with results:\n{task_steps_with_results}")
        return execute_infos
