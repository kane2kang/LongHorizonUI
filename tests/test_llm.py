import json
import os
import pdb
import sys

sys.path.append(".")
from dotenv import load_dotenv

load_dotenv()
import pdb
import time


def test_openai_api():
    """
    测试OPENAI的接口
    :return:
    """
    import os
    from LonghorizonAgent.common import llm_provider, utils

    OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "")
    OPENAI_KEY = os.getenv("OPENAI_KEY", "")
    MODEL = "gpt-4o"

    llm = llm_provider.LLMProvider(llm_provider="openai", model=MODEL, base_url=OPENAI_ENDPOINT, api_key=OPENAI_KEY)

    chat_messages = []
    chat_messages = llm.add_message("system", "you are a helpful AI assistant.", chat_messages)
    image_path = "data/examples/image.png"
    prompt = """
    Please provide a detailed description of the following game screenshot: Include all visible text, icons present, 
    and the information conveyed by each icon. Additionally, identify which page or screen of the game 
    this screenshot represents.
    """
    image_base64 = utils.encode_image(image_path)
    chat_messages = llm.add_message("user", prompt, chat_messages, [image_base64])
    t0 = time.time()
    chat_ret = llm.invoke(chat_messages)
    print(chat_ret)
    print(time.time() - t0)


def test_azure_openai_api():
    """
    测试AZURE OPENAI的接口
    :return:
    """
    import os
    from LonghorizonAgent.common import llm_provider, utils

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    MODEL = "gpt-4o"

    llm = llm_provider.LLMProvider(llm_provider="azure_openai", model=MODEL, base_url=AZURE_OPENAI_ENDPOINT,
                                   api_key=AZURE_OPENAI_API_KEY)

    chat_messages = []
    chat_messages = llm.add_message("system", "you are a helpful AI assistant.", chat_messages)
    image_path = "data/examples/image.png"
    prompt = """
       Please provide a detailed description of the following game screenshot: Include all visible text, icons present, 
       and the information conveyed by each icon. Additionally, identify which page or screen of the game 
       this screenshot represents.
       """
    image_base64 = utils.encode_image(image_path)
    chat_messages = llm.add_message("user", prompt, chat_messages, [image_base64])
    t0 = time.time()
    chat_ret = llm.invoke(chat_messages)
    print(chat_ret)
    print(time.time() - t0)


def test_qwen_api():
    """
    测试QWEN的接口
    :return:
    """
    import os
    from LonghorizonAgent.common import llm_provider, utils

    OPENAI_ENDPOINT = os.getenv("DASHSCOPE_ENDPOINT", "")
    OPENAI_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    MODEL = "qwen-vl-max"

    llm = llm_provider.LLMProvider(llm_provider="openai", model=MODEL, base_url=OPENAI_ENDPOINT, api_key=OPENAI_KEY)

    chat_messages = []
    chat_messages = llm.add_message("system", "you are a helpful AI assistant.", chat_messages)
    image_path = "data/examples/image.png"
    prompt = """
        Please provide a detailed description of the following game screenshot: Include all visible text, icons present, 
        and the information conveyed by each icon. Additionally, identify which page or screen of the game 
        this screenshot represents.
        """
    image_base64 = utils.encode_image(image_path)
    chat_messages = llm.add_message("user", prompt, chat_messages, [image_base64])
    t0 = time.time()
    chat_ret = llm.invoke(chat_messages)
    print(chat_ret)
    print(time.time() - t0)


def test_gemini_api():
    import os
    from LonghorizonAgent.common import llm_provider, utils

    # pip install --upgrade google-genai
    # gcloud auth application-default login

    # model = "gemini-2.0-flash"
    model = "gemini-2.5-pro-preview-05-06"
    project = os.environ.get("GOOGLE_PROJECT", "")
    location = os.environ.get("GOOGLE_LOCATION", "")
    llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location)

    chat_messages = []
    chat_messages = llm.add_message("system", "you are a helpful AI assistant.", chat_messages)
    image_path = "data/examples/dnf.png"
    prompt = """
            Please provide a detailed description of the following game screenshot: Include all visible text, icons present, 
            and the information conveyed by each icon. Additionally, identify which page or screen of the game 
            this screenshot represents.
            """
    image_base64 = utils.encode_image(image_path)
    chat_messages = llm.add_message("user", prompt, chat_messages, [image_base64])
    t0 = time.time()
    chat_ret = llm.invoke(chat_messages)
    print(chat_ret)
    print(time.time() - t0)


def test_llm_to_detect_box():
    import os
    from LonghorizonAgent.common import llm_provider, utils
    import os
    import cv2

    model = "gemini-2.5-pro-preview-05-06"
    # model = "gemini-2.0-flash"
    google_key_json_path = "./assets/google_keys/turinglab-507d7c079329.json"
    project = os.environ.get("GOOGLE_PROJECT", "")
    location = os.environ.get("GOOGLE_LOCATION", "")
    llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location,
                                   google_key_json_path=google_key_json_path)

    # AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    # MODEL = "gpt-4o"
    # llm = llm_provider.LLMProvider(llm_provider="azure_openai", model=MODEL, base_url=AZURE_OPENAI_ENDPOINT,
    #                                api_key=AZURE_OPENAI_API_KEY)

    system_prompt = """
You are an expert UI element locator. Your task is to identify the bounding box of a specific UI element mentioned in the user prompt, based on the provided image.

You MUST output your response *only* as a JSON object containing two keys: "think" and "box".

1.  `think`: Provide a brief step-by-step explanation (in English) of how you identified the element and its bounding box. Focus on visual cues.
2.  `box`: Provide the coordinates of the bounding box as a list of four integer pixel values: `[x_min, y_min, x_max, y_max]`.
    - `x_min`, `y_min`: Coordinates of the top-left corner.
    - `x_max`, `y_max`: Coordinates of the bottom-right corner.

Do not include any introductory phrases, acknowledgements, or any text outside the JSON object itself.
Example Output:
{
  "think": "The 'back' arrow icon is located in the top-left corner of the header bar. I identified its visual shape and position relative to the title.",
  "box": [40, 80, 120, 160]
}
"""
    image_path = "tmp/screenshots/android_2NSDU20827006542/983e31dc-e13a-43b1-86cd-74736a7b433e-draw.png"

    chat_messages = []
    chat_messages = llm.add_message("system", system_prompt, chat_messages)
    user_prompt = "打开排位"
    image_base64 = utils.encode_image(image_path)
    chat_messages = llm.add_message("user", user_prompt, chat_messages, [image_base64])

    response_text = llm.invoke(chat_messages, temperature=0.4)

    if not response_text:
        print("\nError: Received empty response from LLM.")
        return

    box_coords = []  # Changed from center_coords
    try:
        # Clean potential markdown fences ```json ... ```
        cleaned_response = response_text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()

        # Parse the JSON string
        data = json.loads(cleaned_response)
        think_text = data.get("think", "Think key not found.")
        box_coords = data.get("box", [])  # Look for 'box' key

        # Basic validation for bounding box [x_min, y_min, x_max, y_max]
        if not isinstance(box_coords, list) or len(box_coords) != 4:
            print(f"Warning: 'box' data is not a list of 4 numbers: {box_coords}")
            box_coords = []  # Reset if invalid format
        else:
            # Additional check: ensure all elements are numbers (int or float)
            if not all(isinstance(coord, (int, float)) for coord in box_coords):
                print(f"Warning: 'box' coordinates contain non-numeric values: {box_coords}")
                box_coords = []  # Reset if invalid format

        print(f"\nParsed Thinking:\n{think_text}")
        print(f"\nParsed Bounding Box (Normalized 0-1000): {box_coords}")  # Updated print statement

    except json.JSONDecodeError as e:
        print(f"\nError parsing JSON response: {e}")
        print(f"LLM did not return valid JSON. Response was:\n{response_text}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during parsing: {e}")

    image = cv2.imread(image_path)

    if image is None:
        print(f"\nError: Could not load image from path: {image_path}")
        return

    h, w = image.shape[:2]

    if box_coords:
        try:
            # Convert normalized coordinates to actual pixel values
            # Ensure coordinates are integers for drawing
            x_min_norm, y_min_norm, x_max_norm, y_max_norm = box_coords

            x_min = int(x_min_norm / 1000 * w)
            y_min = int(y_min_norm / 1000 * h)
            x_max = int(x_max_norm / 1000 * w)
            y_max = int(y_max_norm / 1000 * h)

            # Define points for the rectangle
            pt1 = (x_min, y_min)  # Top-left corner
            pt2 = (x_max, y_max)  # Bottom-right corner

            color = (0, 0, 255)  # Define the rectangle's color (BGR format - Red)
            thickness = 2  # Rectangle line thickness

            # Draw the rectangle
            cv2.rectangle(image, pt1, pt2, color, thickness)

            print(f"\nDrawing bounding box from {pt1} to {pt2}")

            # --- Display or Save the result ---
            output_image_path = os.path.splitext(image_path)[0] + "-detect-box.png"  # Changed output name
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, image)
            print(f"Result image saved to: {output_image_path}")

        except ValueError:
            print(f"\nError: Bounding box coordinates are not valid numbers: {box_coords}")
        except Exception as e:
            print(f"\nError during drawing or saving image: {e}")
    else:
        print("\nNo valid bounding box found or provided by the LLM.")
        print(response_text)


def test_llm_to_detect_box_with_hints():
    import os
    from LonghorizonAgent.common import llm_provider, utils
    import os
    import cv2

    # model = "gemini-2.5-pro-preview-05-06"
    # # model = "gemini-2.0-flash"
    # google_key_json_path = "./assets/google_keys/turinglab-507d7c079329.json"
    # project = os.environ.get("GOOGLE_PROJECT", "")
    # location = os.environ.get("GOOGLE_LOCATION", "")
    # llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location,
    #                                google_key_json_path=google_key_json_path)

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    MODEL = "gpt-4o"
    llm = llm_provider.LLMProvider(llm_provider="azure_openai", model=MODEL, base_url=AZURE_OPENAI_ENDPOINT,
                                   api_key=AZURE_OPENAI_API_KEY)

    system_prompt = """
You are an expert UI element locator. Your task is to identify the user-specified UI element in the provided image. The image may contain numbered highlighted regions (with indices shown in the top-left corner inside the highlight) corresponding to pre-detected elements, with coordinate hints provided in the user prompt.

Output Instructions:
Your goal is to indicate the location of the target element. You MUST output your response *only* as a JSON object containing two keys: `think` and EITHER `highlight_ind` OR `box`. Choose based on the following logic:

1.  **Check for Direct Match with Highlight:**
    *   Determine if the UI element requested by the user *directly and accurately* corresponds to one of the numbered highlighted regions visible in the image.
    *   **Conditions for using `highlight_ind`:**
        *   The target element is clearly the **primary subject** within the highlighted region.
        *   The highlight's **index number** (e.g., '1', '2') is reasonably visible or clearly associated with that specific element.
        *   The highlighted area provides a **reasonably good fit** for the element (i.e., it's not excessively large encompassing many other elements, and the overlap is high).
    *   If ALL these conditions are met, output the `highlight_ind` key with the corresponding integer index number.

2.  **Fallback to `box` Coordinates:**
    *   Output the `box` key if `highlight_ind` is not appropriate, specifically if:
        *   The target element does **not** correspond to any highlighted region.
        *   The target element *does* correspond to a highlighted region, but the **index number is unclear, cut off, or not visible**.
        *   The target element *does* correspond to a highlighted region, but the highlighted area is **inaccurate or too large** (e.g., it covers much more than the target element, or the element is only a small part of the highlight).
    *   If outputting `box`, provide the *precise* coordinates of the target element's bounding box as a list of four integer pixel values: `[x_min, y_min, x_max, y_max]`. Determine these coordinates carefully, even if refining an inaccurate highlight hint.

3.  **`think` Explanation (Mandatory):**
    *   Always include the `think` key.
    *   Provide a brief step-by-step explanation (in English) of your reasoning.
    *   **If outputting `highlight_ind`**: Explain why you concluded it was a direct, accurate match meeting the criteria (element match, visible index, good fit).
    *   **If outputting `box`**: Explain *why* `highlight_ind` was not suitable (e.g., "target element not highlighted", "highlight 3 exists but number is obscured", "highlight 5 is too large, refining box") and describe how you determined the precise `box` coordinates (e.g., "refining hint coordinates", "locating independently near highlight 2").

**Output Format Summary:**

*   **Scenario 1 (Direct Match):**
    ```json
    {
      "think": "Explanation why highlight_ind is chosen...",
      "highlight_ind": <integer_index>
    }
    ```
*   **Scenario 2 (No Match / Bad Match / Fallback):**
    ```json
    {
      "think": "Explanation why box is chosen and how coordinates were found...",
      "box": [x_min, y_min, x_max, y_max]
    }
    ```

Do not include any introductory phrases, acknowledgements, or any text outside the single JSON object itself.

**Example Output 1 (Using `highlight_ind`):**
```json
{
  "think": "The user wants the 'back' arrow. This element corresponds directly to the highlighted region labeled with the number 1 in the top-left corner. The number is visible and the highlight accurately bounds the arrow.",
  "highlight_ind": 1
}
    """

    image_path = "./tmp/screenshots/android_2NSDU20827006542/e567cfc1-c7cb-44cc-b727-434ba805df05-highlight.png"
    user_prompt = "点击同意协议"

    chat_messages = []
    chat_messages = llm.add_message("system", system_prompt, chat_messages)
    image_base64 = utils.encode_image(image_path)
    chat_messages = llm.add_message("user", user_prompt, chat_messages, [image_base64])
    response_text = llm.invoke(chat_messages, temperature=0.4)

    if not response_text:
        print("\nError: Received empty response from LLM.")
        return

    box_coords = []  # Changed from center_coords
    try:
        # Clean potential markdown fences ```json ... ```
        cleaned_response = response_text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()

        # Parse the JSON string
        data = json.loads(cleaned_response)
        think_text = data.get("think", "Think key not found.")
        box_coords = data.get("box", [])  # Look for 'box' key

        # Basic validation for bounding box [x_min, y_min, x_max, y_max]
        if not isinstance(box_coords, list) or len(box_coords) != 4:
            print(f"Warning: 'box' data is not a list of 4 numbers: {box_coords}")
            box_coords = []  # Reset if invalid format
        else:
            # Additional check: ensure all elements are numbers (int or float)
            if not all(isinstance(coord, (int, float)) for coord in box_coords):
                print(f"Warning: 'box' coordinates contain non-numeric values: {box_coords}")
                box_coords = []  # Reset if invalid format
        print(f"\nParsed Thinking:\n{think_text}")

    except json.JSONDecodeError as e:
        print(f"\nError parsing JSON response: {e}")
        print(f"LLM did not return valid JSON. Response was:\n{response_text}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during parsing: {e}")

    # --- Draw Bounding Box using OpenCV ---
    image = cv2.imread(image_path)

    if image is None:
        print(f"\nError: Could not load image from path: {image_path}")
        return

    h, w = image.shape[:2]

    if box_coords:  # Check if box_coords is valid (non-empty list of 4 numbers)
        try:
            # Convert normalized coordinates to actual pixel values
            # Ensure coordinates are integers for drawing
            x_min_norm, y_min_norm, x_max_norm, y_max_norm = box_coords

            x_min = int(x_min_norm / 1000 * w)
            y_min = int(y_min_norm / 1000 * h)
            x_max = int(x_max_norm / 1000 * w)
            y_max = int(y_max_norm / 1000 * h)

            # Define points for the rectangle
            pt1 = (x_min, y_min)  # Top-left corner
            pt2 = (x_max, y_max)  # Bottom-right corner

            color = (0, 0, 255)  # Define the rectangle's color (BGR format - Red)
            thickness = 2  # Rectangle line thickness

            # Draw the rectangle
            cv2.rectangle(image, pt1, pt2, color, thickness)

            print(f"\nDrawing bounding box from {pt1} to {pt2}")

            # --- Display or Save the result ---
            output_image_path = os.path.splitext(image_path)[0] + "-detect-box.png"  # Changed output name
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, image)
            print(f"Result image saved to: {output_image_path}")

        except ValueError:
            print(f"\nError: Bounding box coordinates are not valid numbers: {box_coords}")
        except Exception as e:
            print(f"\nError during drawing or saving image: {e}")
    else:
        print("\nNo valid bounding box found or provided by the LLM.")
        print(response_text)


def test_llm_to_detect_center():
    import os
    from LonghorizonAgent.common import llm_provider, utils
    import os
    import cv2

    model = "gemini-2.5-pro-preview-05-06"
    # model = "gemini-2.0-flash"
    google_key_json_path = "./assets/google_keys/turinglab-507d7c079329.json"
    project = os.environ.get("GOOGLE_PROJECT", "")
    location = os.environ.get("GOOGLE_LOCATION", "")
    llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location,
                                   google_key_json_path=google_key_json_path)

    # AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    # MODEL = "gpt-4o"
    # llm = llm_provider.LLMProvider(llm_provider="azure_openai", model=MODEL, base_url=AZURE_OPENAI_ENDPOINT,
    #                                api_key=AZURE_OPENAI_API_KEY)

    # --- System Prompt (Define the AI's task and output format) ---
    system_prompt = """
You are an expert UI element locator. Your task is to identify the center point of a specific UI element mentioned in the user prompt, based on the provided image.

You MUST output your response *only* as a JSON object containing two keys: "think" and "center".

1.  `think`: Provide a brief step-by-step explanation (in English) of how you identified the element and its location. Focus on visual cues.
2.  `center`: Provide the coordinates of the center point as a list of two integer pixel values: `[x, y]`. `x` is the horizontal coordinate (from left), and `y` is the vertical coordinate (from top) of the element's approximate center.

Do not include any introductory phrases, acknowledgements, or any text outside the JSON object itself.
"""

    image_path = "./tmp/agent_outputs/AutoExecutionAgent-7c33cee7-7321-4901-887e-58582eff44f6/screenshots/192e7d66-8f43-4640-81f1-3ee5e76abf21.png"
    user_prompt = "点击删除按钮"

    chat_messages = []
    chat_messages = llm.add_message("system", system_prompt, chat_messages)
    image_base64 = utils.encode_image(image_path)
    chat_messages = llm.add_message("user", user_prompt, chat_messages, [image_base64])
    response_text = llm.invoke(chat_messages, temperature=0.4)

    if not response_text:
        print("\nError: Received empty response from LLM.")
        return

    # --- Parse LLM Response ---
    center_coords = []  # Changed from box_coords
    try:
        # Clean potential markdown fences ```json ... ```
        if response_text.strip().startswith("```json"):
            response_text = response_text.strip()[7:-3].strip()
        elif response_text.strip().startswith("```"):
            response_text = response_text.strip()[3:-3].strip()

        # Parse the JSON string
        data = json.loads(response_text)
        think_text = data.get("think", "Think key not found.")
        center_coords = data.get("center", [])  # Look for 'center' key

        # Basic validation for center point [x, y]
        if not isinstance(center_coords, list) or (center_coords and len(center_coords) != 2):
            print(f"Warning: 'center' data is not a list of 2 numbers: {center_coords}")
            center_coords = []  # Reset if invalid format

        print(f"\nParsed Thinking:\n{think_text}")
        print(f"\nParsed Center Coordinates: {center_coords}")  # Updated print statement

    except json.JSONDecodeError as e:
        print(f"\nError parsing JSON response: {e}")
        print("LLM did not return valid JSON.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during parsing: {e}")

    # --- Draw Bounding Box using OpenCV ---
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    if image is None:
        print(f"\nError: Could not load image from path: {image_path}")
        return

    if center_coords:  # Check if center_coords is valid
        try:
            # Ensure coordinates are integers
            x, y = map(int, center_coords)
            x = int(x / 1000 * w)
            y = int(y / 1000 * h)
            center_point = (x, y)
            radius = 10  # Define the circle's radius
            color = (0, 0, 255)  # Define the circle's color (BGR format - Red)
            thickness = 2  # Circle line thickness (use -1 for a filled circle)

            # Draw the circle
            cv2.circle(image, center_point, radius, color, thickness)

            print(f"\nDrawing circle at center: {center_point} with radius {radius}")

            # --- Display or Save the result ---
            output_image_path = os.path.splitext(image_path)[0] + "-detect-center.png"  # Changed output name
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, image)
            print(f"Result image saved to: {output_image_path}")

        except ValueError:
            print("\nError: Center coordinates are not valid numbers.")
        except Exception as e:
            print(f"\nError during drawing or saving image: {e}")
    else:
        print("\nNo valid center point found or provided by the LLM.")


def test_llm_api_grid():
    import os
    from LonghorizonAgent.common import llm_provider, utils
    import os
    import cv2

    # model = "gemini-2.5-pro-preview-05-06"
    # # model = "gemini-2.0-flash"
    # google_key_json_path = "./assets/google_keys/turinglab-507d7c079329.json"
    # project = os.environ.get("GOOGLE_PROJECT", "")
    # location = os.environ.get("GOOGLE_LOCATION", "")
    # llm = llm_provider.LLMProvider(llm_provider="gemini", model=model, project=project, location=location,
    #                                google_key_json_path=google_key_json_path)

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    MODEL = "gpt-4o"
    llm = llm_provider.LLMProvider(llm_provider="azure_openai", model=MODEL, base_url=AZURE_OPENAI_ENDPOINT,
                                   api_key=AZURE_OPENAI_API_KEY)

    system_prompt = """
# Role
You are an AI assistant specialized in pinpointing UI elements within a color-coded grid on an image.

# Grid System Description
The image features a grid with these key elements:
1.  **Colored Grid Lines:** Brightly colored horizontal and vertical lines form the actual grid cell boundaries. 
2.  **Colored Index Markers:** Row/Column indices (numbers) are shown in colored squares at the edges.
3.  **Crucial Color Correlation:** The color of the horizontal **grid line** forming the *bottom border* of a cell **exactly matches** the background color of that cell's corresponding row index marker square. Similarly, the color of the vertical **grid line** forming the *right border* of a cell **exactly matches** the background color of that cell's corresponding column index marker square. This is essential for verifying coordinates.

# Task
Given a description of a UI element, perform these steps:
1.  **Locate Element Center:** Find the geometric center of the specified UI element. Determine the main grid cell (bounded by the colored grid lines) that contains the element's center.
2.  **Determine Cell Indices (`cell_inds`):** Identify the `row_index` and `column_index` for this cell.
3.  **Verify Indices with Color:** **Confirm** the indices by matching the color of the **colored grid line** forming the **bottom border** of the identified cell with the color of the corresponding **row index marker**, AND matching the color of the **colored grid line** forming the **right border** of the cell with the color of the corresponding **column index marker**. This step is mandatory for accuracy.
4.  **Determine Inner Cell Position (`cell_position`):**
    *   Mentally divide the identified cell into a 3x3 subgrid (Top/Middle/Bottom, Left/Middle/Right) relative to the cell's geometric center/midpoint.
    *   Determine the `inner_row_index` (1=Top, 2=Middle, 3=Bottom) based on whether the element center is significantly above, near, or significantly below the horizontal midpoint of the cell.
    *   Determine the `inner_col_index` (1=Left, 2=Middle, 3=Right) based on whether the element center is significantly left of, near, or significantly right of the vertical midpoint of the cell.
5.  **Format Output:** Output a **single JSON object** with exactly three keys:
    *   `think`: A brief string explaining your reasoning, specifically mentioning the identified cell, the color verification using border lines and markers (e.g., "Row 5 confirmed by matching dark blue bottom border line with row 5 marker"), and the basis for the inner position relative to the cell's center (e.g., "center is near the geometric center of the cell").
    *   `cell_inds`: Array `[row_index, column_index]`.
    *   `cell_position`: Array `[inner_row_index, inner_col_index]`.

# Output Requirements
- Output **only** the JSON object. No other text, greetings, or explanations outside the JSON structure.

# Example Output
```json
{
  "think": "Element center is within the cell defined by row 5 and col 3. Row 5 confirmed by matching the bottom border color with the row 5 marker, Col 3 confirmed by matching the right border color with the col 3 marker. Center is near the geometric center of the cell.",
  "cell_inds": [5, 3],
  "cell_position": [2, 2]
}
"""
    image_path = "./tmp/screenshots/android_2NSDU20827006542/8eabc4db-c2f0-4f51-a9e2-3b35ee75d594-highlight.png"
    user_prompt = "关闭弹窗"

    chat_messages = []
    chat_messages = llm.add_message("system", system_prompt, chat_messages)
    image_base64 = utils.encode_image(image_path)
    chat_messages = llm.add_message("user", user_prompt, chat_messages, [image_base64])
    response_text = llm.invoke(chat_messages, temperature=0.4)

    if not response_text:
        print("\nError: Received empty response from LLM.")
        return
    print(response_text)


if __name__ == '__main__':
    # test_openai_api()
    # test_azure_openai_api()
    # test_qwen_api()
    # test_gemini_api()
    # test_llm_to_detect_box()
    # test_llm_to_detect_box_with_hints()
    test_llm_to_detect_center()
    # test_llm_api_grid()
