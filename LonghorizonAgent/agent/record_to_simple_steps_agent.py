import pdb
import cv2
import os
import json
import logging
import copy
import math
# Assuming these are your project's modules
from LonghorizonAgent.perception.screen_perception import ScreenPerception
from ..common.llm_provider import LLMProvider
from ..common import llm_provider  # Assuming this is how you import LLMProvider class if needed elsewhere
from ..common import utils

# memory
from LonghorizonAgent.memory.operation_graph import OperationGraph, ActionEdge
from LonghorizonAgent.agent.graph_node import GraphNode
from LonghorizonAgent.common import constants

logger = logging.getLogger(__name__)


class RecordToSimpleStepsAgent:
    """
    将用户录制的原始操作（坐标、截图）转换为与设备无关的、
    基于UI元素和相对位置的自然语言步骤描述。
    Utilizes LLM to interpret actions based on before/after screenshots
    and optional visual hints (red dot for tap/press, green arrow for swipe/drag).
    """

    def __init__(self, llm: LLMProvider, **kwargs):
        """
        Initializes the agent.

        Args:
            llm (LLMProvider): An instance of the LLM provider client.
            **kwargs: Optional keyword arguments.
                system_prompt (str): The system prompt for the LLM.
        """
        self.llm = llm
        self.system_prompt = kwargs.get("system_prompt",
                                        "You are an expert mobile QA automation engineer analyzing user interactions on a mobile screen. Your goal is to describe each user action in clear, concise, natural language, focusing on *what* was interacted with and *how*, independent of screen coordinates or size. This description should be understandable and reproducible by another AI agent on potentially different devices.")
        self.screen_p = ScreenPerception(use_icon_caption=False,
                                         use_rec=False
                                         )
        logger.info("RecordToSimpleStepsAgent initialized.")

    def get_user_prompt(self, action_info, language="Chinese"):
        """
        Generates the user prompt for the LLM to describe a single action step.

        Args:
            action_info (str): JSON string containing raw action data (type, coordinates).
            language (str): The desired output language for the description.

        Returns:
            str: The formatted prompt string for the LLM.
        """

        prompt = f"""
You are given with two images and action information:
1.  'The first image': Shows the screen state the action was performed.
    *   **Visual Hints (Optional):** This image *might* contain a visual hint to help you locate the action area.
        *   If present, a **red rectangle or red dot** indicates the approximate location of a Tap or Press action.
        *   If present, a **green arrow** indicates the path and direction of a Swipe or Drag action.
    *   **Reference Only:** If a red rectangle or red dot or green arrow *is* present, use it ONLY as a reference to identify the target. **Do NOT mention the 'a red rectangle or red dot' or 'green arrow' in your final action description.**
    *   Analyze along with the 'Action information' (type and coordinates) provided to understand the action type and location.
2.  'The second image': Shows the screen state after the action was performed.
    *   Used to help you verify whether the action is correct.
    *   Help you output accurate action results.
3.  [Third Image (Optional)]: Zoomed-in view of the operation area (cropped from the first image).
    *   **Critical for Precision:** 
        - Focus on this image to identify small UI elements (e.g., icons, text).
        - Use details from this image to resolve ambiguous elements (e.g., buttons with similar text).
4.  'Action information': Raw data about the action performed (e.g., type, coordinates). This tells you the type of action (Tap, Swipe, etc.) and where it occurred.
5   SPECIAL INSTRUCTION FOR NetCap APP:** When you detect that the current application is NetCap, immediately:
        1. Check the screen for any visible UUID string (typically a 36-character string in the format dc801926-d52b-4531-b353-16a5d9f45529)
        2. If found, extract the UUID and output it in the "action_result" field with the prefix "检测到UUID: "
    ```json
    {action_info}
    ```

Your task is to generate a JSON response describing the action performed *on* the 'Screenshot':

Instructions for the "action" description:
*   **Identify the Target Element:** Based on your analysis of the first image (using the hint), describe the specific UI element the user interacted with (e.g., button, icon, text field, image, list item, specific text label).
*   **Use Precise Verbs:** Use clear action verbs based on the 'Action information' type (e.g., "tapped", "pressed", "swiped [direction, e.g., left/right/up/down]", "dragged the [element description] [direction]").
*   **Describe Location Relatively:** Use relative positioning based on visual landmarks *visible in the 'Screenshot'* (e.g., "in the top-right corner", "below the title 'Settings'", "the third button from the left"). **Avoid absolute coordinates (e.g., (100, 200)) or pixel-based descriptions in your output.**
*   **Handle Duplicate Text:** If the same text appears multiple times on the 'Screenshot', provide context using relative location or surrounding elements visible in the 'Screenshot' (e.g., "tapped the 'Details' link under the 'Product Info' section").
*   **Avoid Color Interference from Overlays:** Analysis overlays (like red rectangles, red dots, or green arrows) might cover parts of the UI element. Describe the element based on its inherent characteristics (text, icon type, shape, relative position) and avoid using colors like 'red' or 'green' in your description.
*   **Handle Special Icons:** Pay attention to these specific icons if they appear:
    *   **NetCap Floating Button:** If the target is a distinctive **floating circular button filled with blue color**. If you find a red box or a red dot on this icon, return to this action description directly: `点击蓝色悬浮按钮，返回NetCap应用`.
    *   **System Navigation (Visuals):** Some phone screen edges might show system navigation buttons visually (返回首页(Home): 通常是一个房子的形状; 查看或切换最近应用(Recent): 通常是三条横线; 返回(Back): 通常是弯曲箭头). **Do NOT describe the shape** of these icons. Instead, use the functional description outlined in the 'Handle System Actions' point below.
*   **Handle System Actions:** If the action indicates a system action (like Home, Back, Recent), describe it simply as:
        * Home: 返回首页
        * Back: 返回上一页
        * Recent: 查看或切换最近应用

Please output in {language}. Respond ONLY in the following JSON format:
```json
# more sample
{{
    "action": "A very concise description of the action performed (e.g., 'tapped [element]', 'swiped left on [element]', 'dragged [element] to [position]'). Only describe the action type and the target element. Do not include the result of the action.",
    "action_result": "A very brief phrase describing the outcome (e.g., 'new page opened', 'popup closed').",
}}

# more details
# {{
#     "action": "A concise, natural language description of the action performed according to the rules above.",
#     "action_result": "A detailed description of the result of the action, such as navigation to a different page (with detailed information about this page) or triggering a function or popup. Reference the second image for more context.",
# }}
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

    def execute(self, record_dir, task_language="Chinese", **kwargs):
        """
        Executes the process of converting recorded actions to natural language steps.

        Args:
            record_dir (str): Path to the directory containing 'actions.json' and 'screenshot/'.
            task_language (str): Language for the final step descriptions and task name (default: "Chinese").
            **kwargs: Optional keyword arguments.
                temperature (float): Temperature setting for LLM generation (default: 0.5).

        Returns:
            list: A list of dictionaries, each containing the generated action description
                  and the original raw action data. Returns empty list on critical error.
        """

        execute_infos = []
        output_dir = os.path.join(record_dir, "screenshot_draw")
        action_json_file = os.path.join(record_dir, "actions.json")
        screenshot_dir = os.path.join(record_dir, "screenshot")

        if not os.path.exists(action_json_file):
            logger.error(f"Action file not found: {action_json_file}")
            return {}
        if not os.path.exists(screenshot_dir):
            logger.error(f"Screenshot directory not found: {screenshot_dir}")
            return {}

        os.makedirs(output_dir, exist_ok=True)

        try:
            with open(action_json_file, "r", encoding="utf-8") as fin:
                action_infos = json.load(fin)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {action_json_file}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to read action file {action_json_file}: {e}")
            return {}

        # memory_graph
        constants.GRAPH_DIR = record_dir
        constants.GRAPH_NAME = "operation_ui_graph"
        op_graph = OperationGraph(graph_dir=record_dir, graph_name="operation_ui_graph")
        # prev_node: GraphNode | None = None

        num_actions = len(action_infos)

        for i, action_data in enumerate(action_infos):
            step_num = i + 1
            logger.info(f"Processing action {step_num}/{num_actions}...")
            # Initialize conversation history for LLM
            chat_messages = []
            chat_messages = self.llm.add_message("system", self.system_prompt, chat_messages)
            try:
                # --- Image Handling ---
                if "screenshot" not in action_data:
                    logger.warning(f"Action {step_num} missing 'screenshot' key. Skipping.")
                    continue

                # Normalize path and construct full path for the 'before' image
                screenshot_path = os.path.normpath(action_data["screenshot"])
                screenshot_path = os.path.normpath(screenshot_path).replace('\\', '/')
                image_path = os.path.join(record_dir, "screenshot", os.path.basename(screenshot_path))
                image_index = int(os.path.splitext(os.path.basename(image_path))[0])

                if not os.path.exists(image_path):
                    logger.warning(f"Screenshot file not found for action {step_num}: {image_path}. Skipping.")
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Failed to load image: {image_path}. Skipping action {step_num}.")
                    continue
                img_h, img_w = image.shape[:2]

                screen_status = self.screen_p.run_perception(image)
                screen_perception_info = screen_status.perception_info

                # graph if same screenshot is find many times, we can consider to use key as the cache
                cur_node = GraphNode(
                    screenshot_img_path=image_path,
                    perception_infos=screen_perception_info,
                    description=f"Step-{step_num}"
                )
                op_graph.add_node(cur_node)

                # 生成节点名
                custom_node_name = f"step_{step_num:03d}"
                new_rel_path = os.path.join("images", f"{custom_node_name}.png")
                new_abs_path = os.path.join(record_dir, "operation_ui_graph", new_rel_path)

                old_rel_path = cur_node.image_path
                old_abs_path = os.path.join(record_dir, "operation_ui_graph", old_rel_path)
                try:
                    if os.path.exists(old_abs_path) and not os.path.exists(new_abs_path):
                        os.rename(old_abs_path, new_abs_path)
                except OSError as e:
                    logger.warning(f"Rename image failed ({old_abs_path} → {new_abs_path}): {e}")

                # 更新节点字段 & op_graph.nodes 索引
                old_node_id = cur_node.node_name  # uuid
                if old_node_id != custom_node_name:
                    cur_node.node_name = custom_node_name
                    cur_node.image_path = new_rel_path

                    if old_node_id in op_graph.nodes:
                        del op_graph.nodes[old_node_id]
                    op_graph.nodes[custom_node_name] = cur_node

                action_data["screenshot_width"] = img_w
                action_data["screenshot_height"] = img_h

                # Action Filtering
                action_data = self.filter_action(copy.deepcopy(action_data))  # Filter a copy

                # Visualization (Optional Hints)
                image_draw = image.copy()

                hint_drawn = False
                bbox = None  # save bbox
                cropped_encoded = None


                if "action" in action_data:
                    action_str = action_data["action"]
                    try:
                        if action_str.startswith("Tap") or action_str.startswith("Press"):

                            # 解析点击坐标
                            coords_str = action_str.split("(")[1].split(")")[0]
                            tap_x, tap_y = map(int, [c.strip() for c in coords_str.split(",")])
                            candidate_boxes = []

                            for pinfo_ in screen_perception_info:
                                x1, y1, x2, y2 = pinfo_["box"]

                                # 检查点击坐标是否在当前box内
                                if tap_x >= x1 and tap_x <= x2 and tap_y >= y1 and tap_y <= y2:
                                    area = (x2 - x1) * (y2 - y1)
                                    candidate_boxes.append((area, pinfo_["box"]))

                            # select area最大的box
                            selected_box = None
                            if candidate_boxes:
                                # area降序排序
                                candidate_boxes.sort(key=lambda x: x[0], reverse=True)
                                selected_box = candidate_boxes[0][1]

                            cv2.circle(image_draw, (tap_x, tap_y), 10, (0, 0, 255), -1)
                            if selected_box is not None:
                                x1, y1, x2, y2 = selected_box
                                bbox = [x1, y1, x2, y2]
                                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 0, 255), 10)

                                crop_dir = os.path.join(record_dir, "crop")
                                os.makedirs(crop_dir, exist_ok=True)

                                crop_filename = f"step_{step_num:03d}_crop_{image_index:03d}.png"
                                crop_path = os.path.join(crop_dir, crop_filename)

                                # 检查坐标
                                if (x1 >= x2) or (y1 >= y2) or (x2 > img_w) or (y2 > img_h):
                                    logger.warning(f"Invalid closest_box coordinates: {(x1, y1, x2, y2)}")
                                else:
                                    # crop and save
                                    cropped_region = image[y1:y2, x1:x2]
                                    if cropped_region.size > 0:
                                        success = cv2.imwrite(crop_path, cropped_region)
                                        if success:
                                            cropped_encoded = utils.encode_image(crop_path)
                                            if not cropped_encoded:
                                                logger.warning("Failed to encode cropped region")
                                        else:
                                            logger.error(f"Failed to save crop image: {crop_path}")
                                    else:
                                        logger.warning("Empty cropped region")
                            else:
                                # 创建一个点击点周围的小矩形
                                pad = min(img_w, img_h) // 20
                                x1 = max(0, tap_x - pad)
                                y1 = max(0, tap_y - pad)
                                x2 = min(img_w - 1, tap_x + pad)
                                y2 = min(img_h - 1, tap_y + pad)
                                bbox = [x1, y1, x2, y2]
                                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 0, 255), 10)

                            # ICON 坐标计算
                            pad = 10
                            x1_icon = max(0, tap_x - pad)
                            y1_icon = max(0, tap_y - pad)
                            x2_icon = min(img_w - 1, tap_x + pad)
                            y2_icon = min(img_h - 1, tap_y + pad)

                            cur_node.add_icon(
                                icon_box=(x1_icon, y1_icon, x2_icon, y2_icon),
                                icon_description="tap-point",
                                clickable=True,
                                is_clicked=True
                            )

                            hint_drawn = True


                        elif action_str.startswith("Swipe") or action_str.startswith("Drag"):

                            coords_str1 = action_str.split("(")[1].split(")")[0]
                            x1, y1 = map(int, [c.strip() for c in coords_str1.split(",")])
                            coords_str2 = action_str.split("(")[2].split(")")[0]
                            x2, y2 = map(int, [c.strip() for c in coords_str2.split(",")])
                            # Clamp coordinates to image bounds
                            x1, y1 = max(0, min(img_w - 1, x1)), max(0, min(img_h - 1, y1))
                            x2, y2 = max(0, min(img_w - 1, x2)), max(0, min(img_h - 1, y2))

                            # 存储bbox
                            bbox = [
                                min(x1, x2),
                                min(y1, y2),
                                max(x1, x2),
                                max(y1, y2)
                            ]
                            # Draw green arrow
                            cv2.arrowedLine(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 10,
                                            tipLength=0.3)  # Green arrow

                            hint_drawn = True

                    except (IndexError, ValueError, Exception) as e:
                        logger.error(f"Error drawing hint for action '{action_str}': {e}", exc_info=True)
                        bbox = [0, 0, img_w, img_h]

                # 确保每个操作都有bbox
                if bbox is None:
                    if "bbox" in action_data:
                        bbox = action_data["bbox"]
                    else:
                        bbox = [0, 0, img_w, img_h]

                action_data["bbox"] = bbox


                output_image_path = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_image_path, image_draw)
                logger.debug(
                    f"Saved {'hinted' if hint_drawn else 'original'} image for step {step_num} to {output_image_path}")

                # --- Prepare 'After' Image ---
                # Use the *next* action's screenshot as the 'after' image for the current action
                # Handle the last action case where there is no next screenshot
                next_image_index = image_index + 1
                next_image_basename = f"{next_image_index:03d}.png"  # Assuming 3-digit padding like 001.png
                next_image_path = os.path.join(screenshot_dir, next_image_basename)
                nex_image_encoded = None
                if os.path.exists(next_image_path):
                    nex_image_encoded = utils.encode_image(next_image_path)
                    if nex_image_encoded is None:
                        logger.warning(f"Failed to encode 'after' image: {next_image_path}")
                else:
                    logger.warning(
                        f"'After' image not found for step {step_num}: {next_image_path}. Proceeding without it.")

                # --- Prepare Data for LLM ---
                # Create a clean version of action_info for the prompt
                action_info_for_prompt = copy.deepcopy(action_data)
                action_info_for_prompt.pop("screenshot", None)
                action_info_str = json.dumps(action_info_for_prompt, indent=2)

                user_prompt = self.get_user_prompt(action_info_str, task_language)

                # Encode the 'before' image (potentially with hint)
                cur_image_encoded = utils.encode_image(output_image_path)

                if cur_image_encoded is None:
                    logger.error(f"Failed to encode 'before' image: {output_image_path}. Skipping step {step_num}.")
                    continue

                # Build image list for LLM
                image_list = [cur_image_encoded]
                if nex_image_encoded:
                    image_list.append(nex_image_encoded)

                # 加入crop区域
                if cropped_encoded:
                    image_list.append(cropped_encoded)
                    logger.debug(f"Added cropped region to LLM input (size: {cropped_region.shape})")

                # --- Call LLM for Step Description ---
                logger.debug(f"Sending prompt for step {step_num} to LLM.")
                # Add user message for this step
                chat_messages = self.llm.add_message("user", user_prompt, chat_messages, image_list)

                output_content = self.llm.invoke(chat_messages=chat_messages,
                                                 temperature=kwargs.get("temperature", 0.5))
                # --- Process LLM Response ---
                try:
                    # Clean potential markdown code fences
                    action_text = output_content.strip().removeprefix("```json").removesuffix("```").strip()
                    if action_text.endswith(','):
                        action_text = action_text[:-1] + '}'
                    try:
                        exec_info = json.loads(action_text)
                    except json.JSONDecodeError:
                        exec_info = {
                            "action": f"步骤{step_num} - JSON解析失败",
                            "action_result": action_text  # 保留原始响应
                        }

                    exec_info["step_id"] = f"{step_num:03d}"
                    if "action" not in exec_info:
                        raise ValueError("LLM response missing 'action' key.")


                    # Store original action data alongside LLM description
                    action_data["screenshot"] = os.path.join("screenshot", os.path.basename(screenshot_path))
                    exec_info["raw_action"] = action_data
                    execute_infos.append(exec_info)
                    logger.info(f"Step {step_num} action: {exec_info['action']}")
                    logger.info(f"Step {step_num} action result: {exec_info['action_result']}")

                    if prev_node is not None and exec_info is not None:
                        edge = ActionEdge(src_node=prev_node,
                                          dst_node=cur_node,
                                          exec_info=exec_info)
                        op_graph.add_edge(edge)
                    prev_node = cur_node




                except (json.JSONDecodeError, ValueError, Exception) as e:
                    logger.error(f"Failed to parse LLM response for step {step_num}: {e}. Response: '{output_content}'",
                                 exc_info=True)
                    # Optionally save the raw response and skip adding to execute_infos
                    # Or create a placeholder error entry
                    execute_infos.append({
                        "step_id": f"{step_num:03d}",
                        "action": f"Error: Failed to process LLM response for this step.",
                        "error_details": str(e),
                        "raw_llm_response": output_content,
                        "raw_action": action_data
                    })

            except Exception as e:
                logger.error(
                    f"An unexpected error occurred processing action {step_num} ({action_data.get('screenshot', 'N/A')}): {e}",
                    exc_info=True)
                # Decide whether to break or continue
                # break # Stop processing further steps on error
                # Or add an error entry and continue
                execute_infos.append({
                    "step_id": f"{step_num:03d}",
                    "action": f"Error: Unexpected error processing step {step_num}.",
                    "error_details": str(e),
                    "raw_action": action_data
                })
                continue  # Continue to next step

        # --- Generate Task Name ---
        task_name = "Error: Could not generate task name."  # Default value
        if not execute_infos:
            logger.warning("No steps were successfully processed. Cannot generate task name.")
        else:
            logger.info("Generating task name...")
            # Use the refined prompt asking for a concise summary based on generated steps
            # --- Log Summary ---
            task_steps_summary = ""
            for i, exe_info in enumerate(execute_infos):
                # Check if 'action' exists and is not an error message before appending
                action_desc = exe_info.get("action", "N/A")
                if "Error:" in action_desc:
                    task_steps_summary += f"{i + 1}. {action_desc} (Raw: {exe_info.get('raw_action', {}).get('action', 'N/A')})\n"
                else:
                    task_steps_summary += f"{i + 1}. {action_desc}\n"
            task_name_prompt = f"""
            Based on all the AI-generated action infos, please generate a detailed summary for the entire workflow.
            Directly output only the task name or summary text, without any additional explanations or labels.
            Please provide the output in {task_language}.
            """
            chat_messages = self.llm.add_message("system", task_name_prompt, [])
            chat_messages = self.llm.add_message("user", f"Task steps:\n{task_steps_summary}", chat_messages)
            try:
                task_name = self.llm.invoke(chat_messages=chat_messages, temperature=kwargs.get("temperature", 0.5))
                task_name = task_name.strip()  # Clean whitespace
                logger.info(f"Generated Task Name: {task_name}")
            except Exception as e:
                logger.error(f"Failed to generate task name from LLM: {e}", exc_info=True)
                # Keep the default error message for task_name

                # Save Results

            op_graph.save()
            save_json_path = os.path.join(record_dir, "task_infos.json")
            task_infos = None
            try:
                task_infos = {
                    "task_name": task_name,
                    "task_steps": execute_infos
                }
                with open(save_json_path, "w", encoding="utf-8") as fw:
                    json.dump(task_infos, fw, indent=2,
                              ensure_ascii=False)  # Use ensure_ascii=False for non-English chars
                logger.info(f"Saved step info and task name to {save_json_path}")
                logger.info(f"--- Summary ---")
                logger.info(f"Task Name:\n{task_name}")
                logger.info(f"Task Steps:\n{task_steps_summary.strip()}")
                logger.info(f"---------------")
            except Exception as e:
                logger.error(f"Failed to save results to {save_json_path}: {e}", exc_info=True)
            finally:
                return task_infos
