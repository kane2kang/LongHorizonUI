import gradio as gr
import argparse
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import adbutils
# Assume these imports work correctly based on your project structure
from LonghorizonAgent.common import llm_provider, utils
from LonghorizonAgent.agent.auto_execution_agent import AutoExecutionAgent, AutoExecutionConfig
from LonghorizonAgent.agent.record_to_simple_steps_agent import RecordToSimpleStepsAgent
from LonghorizonAgent.system.android_context import AndroidContext, AndroidContextConfig
from LonghorizonAgent.controller.android_controller import AndroidController
from LonghorizonAgent.prompts.auto_execution_prompt import AndroidExecSystemPrompt, AutoExecAgentPrompt
from LonghorizonAgent.common.llm_provider import LLMProvider
import threading  # Import threading
import time  # Import time for sleep

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global state management
ui_state = {
    "agent": None,
    "chatbot_history": [],
    "history_prev_len": 0,  # Track previous length
    "agent_thread": None,  # To hold the agent thread reference
    "output_dir": None,  # Store output dir from agent run
    "agent_id": None,  # Store agent id from agent run
}


def _run_agent_task(agent: AutoExecutionAgent, task: str, task_steps: Optional[str], task_infos: Optional[str]):
    """Target function for the agent thread."""
    global ui_state
    try:
        logging.info(f"Agent thread started for task: {task}")
        # Store the output dir and agent_id in the global state upon completion
        ui_state["output_dir"] = agent.run(task, task_steps=task_steps, task_infos=task_infos)
        ui_state["agent_id"] = agent.agent_id  # Assuming agent stores its ID here after run
        logging.info(f"Agent thread finished. Output directory: {ui_state['output_dir']}")
    except Exception as e:
        logging.error(f"Agent run failed within thread: {e}", exc_info=True)
        error_message = f"**Error:** Agent run encountered an error within its thread.\n\n`{e}`"
        # Append error directly to history for the main thread to pick up
        ui_state["chatbot_history"].append({"role": "assistant", "content": error_message})
    finally:
        # Signal completion implicitly by the thread ending
        pass


def step_output_callback(step_data: Dict[str, Any]):
    """
    Callback function passed to the agent, formats output for gr.Chatbot.
    Appends messages to the global chatbot_history state.
    (Corrected formatting and added base64 image handling)
    """
    step = step_data.get("step", "N/A")
    phase = step_data.get("phase", "N/A")
    screenshot_b64 = step_data.get("screenshot")  # Base64 encoded screenshot string or None

    message_content = ""
    img_html = ""

    if screenshot_b64:
        # Embed screenshot directly in the message
        img_html = f'<img src="data:image/png;base64,{screenshot_b64}" style="max-width:300px; max-height:600px; object-fit:contain; margin:10px 0;" alt="Step {step} Screenshot">'
    else:
        img_html = ""

    if phase == "observation":
        message_content = f"### Step {step} - Observation\n"
        message_content += "**Status:** Observing screen and highlighting elements.\n"  # Add newline
    elif phase == "action":
        message_content = f"### Step {step} - LLM Output & Action\n"  # Add newline
        model_output = step_data.get("model_output", {})
        action_result = step_data.get("action_result", {})
        step_time = step_data.get("step_time", 0)

        eval_prev_goal = model_output.get('evaluation_prev_goal', 'N/A')
        import_contents = model_output.get('import_contents', 'N/A')
        think_process = model_output.get('think', 'N/A')
        next_goal = model_output.get('next_goal', 'N/A')
        action_data = model_output.get('action', {})  # Action data can be complex
        action_str = str(action_data)  # Simple string representation for now
        action_status = 'Success' if not action_result.get('error', None) else 'Failed'
        error_msg = action_result.get("error")

        # Use Markdown for better structure within the chat bubble
        message_content += f"""
*   **Step Cost Time:** `{step_time}`
*   **Eval Previous Goal:** `{eval_prev_goal}`
*   **Important Contents:** `{import_contents}`
*   **Thought Process:** `{think_process}`
*   **Next Goal:** `{next_goal}`
*   **Action:** `{action_str}`
*   **Action Status:** {action_status}
"""  # Corrected f-string usage and formatting
        if error_msg:
            message_content += f"*   **Error:** ```{error_msg}```\n"  # Add newline and code block
    else:
        # Handle unexpected phase gracefully
        message_content = f"### Step {step} - Phase: {phase}\n"
        message_content += f"```\n{step_data}\n```\n"  # Print raw data if phase unknown

    message_content += "\n" + img_html + "\n"  # Add space around image

    # Append the message to the chatbot history
    ui_state["chatbot_history"].append({"role": f"assistant", "content": message_content})
    logging.debug(f"Step {step} - {phase} added to chat history.")


def create_agent(platform="android", llm_name="gemini", llm_model_name="gemini-2.5-pro-preview-05-06"):
    """
    Creates and configures the AutoExecutionAgent instance.
    (Minor corrections and logging)
    """
    logging.info(f"Creating agent for platform: {platform}, LLM: {llm_name}, Model: {llm_model_name}")

    # LLM setup
    if llm_name == "gemini":
        project = os.getenv("LLM_PROJECT")
        location = os.getenv("LLM_LOCATION")

        if not project or not location:
            logging.warning(
                "LLM_PROJECT or LLM_LOCATION not explicitly set. LLM Provider might use defaults or other auth methods.")

        logging.info(f"Attempting to use Gemini Project: {project}, Location: {location}")
        try:
            llm = llm_provider.LLMProvider(
                llm_provider=llm_name,
                model=llm_model_name,
                project=project,
                location=location
            )
        except Exception as e:
            logging.error(f"Failed to initialize LLMProvider: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize LLM Provider. Check credentials and configuration. Error: {e}")

    else:
        raise NotImplementedError(f"LLM provider '{llm_name}' not implemented.")

    # Platform-specific setup
    if platform == "android":
        try:
            devices = adbutils.adb.device_list()
        except Exception as e:
            logging.error(f"ADB command failed. Is ADB installed and in PATH? Error: {e}", exc_info=True)
            raise ValueError(
                f"Failed to list ADB devices. Ensure ADB is installed, running, and authorized. Error: {e}")

        if not devices:
            raise ValueError(
                "No Android devices found. Please connect a device, ensure it's authorized, and ADB is working.")

        android_device_serial = devices[0].serial  # Use .serial attribute
        if not android_device_serial:
            # This case is less likely if device_list() returned a device object
            raise ValueError(f"Could not get serial number for device: {devices[0]}")
        logging.info(f"Using Android device: {android_device_serial}")

        context_config = AndroidContextConfig(
            device_id=android_device_serial,
            use_perception=True,
            use_ocr=True,
            use_ocr_rec=False,
            perception_description_type="normal",
            highlight_type="normal",
            detect_split_x=2,
            detect_split_y=2,
        )
        system_context = AndroidContext(config=context_config)
        controller = AndroidController(highlight_action=True)
        system_prompt_class = AndroidExecSystemPrompt
        agent_prompt_class = AutoExecAgentPrompt
    else:
        raise NotImplementedError(f"Platform '{platform}' not implemented.")

    # Agent configuration
    agent_config = AutoExecutionConfig(
        max_steps=100,
        llm_temperature=0.4,
        keep_last_n_states=2,
    )

    # Create agent instance with the callback
    agent = AutoExecutionAgent(
        agent_config=agent_config,
        controller=controller,
        system_context=system_context,
        llm=llm,
        system_prompt_class=system_prompt_class,
        agent_prompt_class=agent_prompt_class,
        step_output_callback=step_output_callback
    )
    logging.info("Agent created successfully.")
    return agent


def click_to_run_agent(task: str, task_steps: Optional[str], task_infos: Optional[str]):
    """
    Handles the 'Run Agent' button click. Clears history, creates/gets agent,
    runs the agent in a separate thread, and yields updates to the Chatbot UI
    as new messages arrive. Finally yields results (GIF, JSON) and button states.

    Args:
        task: The main task description.
        task_steps: Optional step-by-step instructions.
        task_infos: Optional additional context.

    Yields:
        Updates for Gradio components (Chatbot history, GIF path, JSON path, Button states).
    """
    global ui_state
    logging.info("Run Agent button clicked.")

    # 1. Clear previous history and reset state
    ui_state["chatbot_history"] = []
    ui_state["history_prev_len"] = 0
    ui_state["agent_thread"] = None
    ui_state["output_dir"] = None
    ui_state["agent_id"] = None

    # 2. Immediately yield to clear UI and update buttons
    # Yield empty chat history, None for files, disable Run, enable Stop
    yield [], None, None, gr.update(interactive=False), gr.update(interactive=True)
    time.sleep(0.1)  # Small delay to allow UI to update

    # 3. Construct and add the user's task input as a USER message
    user_input_summary = f"**Task:**\n {task}\n\n"
    if task_steps:
        user_input_summary += f"**Provided Steps:**\n{task_steps}\n\n"
    if task_infos:
        user_input_summary += f"**Additional Info:**\n{task_infos}\n\n"
    # Append as USER message
    ui_state["chatbot_history"].append({"role": "user", "content": user_input_summary})
    ui_state["history_prev_len"] = len(ui_state["chatbot_history"])  # Update prev len

    # 4. Yield AGAIN to show the user message in the chatbot immediately
    yield ui_state["chatbot_history"], None, None, gr.update(interactive=False), gr.update(interactive=True)

    # 5. Get or create the agent instance
    agent = None
    try:
        # Reuse global agent if exists, else create.
        if ui_state["agent"] is None:
            logging.info("No existing agent found, creating a new one.")
            ui_state["agent"] = create_agent()  # Using default params
        agent = ui_state["agent"]
        # Optional: Reset agent's internal state if necessary before a new run
        # agent.reset() # If your agent class has a reset method

    except Exception as e:
        logging.error(f"Failed to create or get agent: {e}", exc_info=True)
        error_message = f"**Error:** Failed to initialize agent.\nIs a device connected and ADB working?\nCheck environment variables (e.g., LLM_PROJECT, LLM_LOCATION, LLM_API_KEY).\n\n`{e}`"
        ui_state["chatbot_history"].append({"role": "assistant", "content": error_message})
        # Update UI with error and reset buttons
        yield ui_state["chatbot_history"], None, None, gr.update(interactive=True), gr.update(interactive=False)
        return  # Stop execution

    # 6. Run the agent's task in a separate thread
    try:
        logging.info(f"Starting agent thread with task: {task}")
        # Create and start the thread
        ui_state["agent_thread"] = threading.Thread(
            target=_run_agent_task,
            args=(agent, task, task_steps, task_infos),
            daemon=True  # Allows main program to exit even if thread is running
        )
        ui_state["agent_thread"].start()

        # 7. Monitor the history and yield updates
        while ui_state["agent_thread"] and ui_state["agent_thread"].is_alive():
            current_len = len(ui_state["chatbot_history"])
            if current_len > ui_state["history_prev_len"]:
                # New messages detected, yield update for chatbot only
                ui_state["history_prev_len"] = current_len
                yield ui_state["chatbot_history"], None, None, gr.update(interactive=False), gr.update(interactive=True)
            time.sleep(0.2)  # Check periodically, adjust sleep time as needed

        # Ensure thread has fully finished before proceeding
        if ui_state["agent_thread"]:
            ui_state["agent_thread"].join(timeout=5.0)  # Wait briefly for thread cleanup
            if ui_state["agent_thread"].is_alive():
                logging.warning("Agent thread did not terminate cleanly after run.")

        # Check if agent run added an error message at the end
        current_len = len(ui_state["chatbot_history"])
        if current_len > ui_state["history_prev_len"]:
            yield ui_state["chatbot_history"], None, None, gr.update(interactive=False), gr.update(interactive=True)
            ui_state["history_prev_len"] = current_len

        logging.info("Agent thread has finished or stop was requested.")

        # 8. Construct final results paths (use state variables filled by thread)
        gif_path = None
        history_json = None
        if ui_state["output_dir"] and ui_state["agent_id"]:
            output_dir = ui_state["output_dir"]
            agent_id = ui_state["agent_id"]
            gif_path = os.path.join(output_dir, f"{agent_id}.gif")
            history_json = os.path.join(output_dir, f"{agent_id}.json")
            logging.info(f"Looking for results: GIF='{gif_path}', JSON='{history_json}'")
            # Check if files actually exist, provide None if not
            if not os.path.exists(gif_path):
                logging.warning(f"Result GIF not found at: {gif_path}")
                gif_path = None
            if not os.path.exists(history_json):
                logging.warning(f"Result JSON not found at: {history_json}")
                history_json = None
        else:
            logging.warning(
                "Agent run did not provide output directory or agent ID via ui_state. Cannot locate results.")

        # 9. Yield final results and reset button states
        logging.info("Yielding final results and resetting buttons.")
        yield ui_state["chatbot_history"], gif_path, history_json, gr.update(interactive=True), gr.update(
            interactive=False)

    except Exception as e:
        # Handle errors occurring in the main thread (e.g., thread creation failure)
        logging.error(f"Error during agent run orchestration: {e}", exc_info=True)
        error_message = f"**Error:** An unexpected error occurred while managing the agent run.\n\n`{e}`"
        ui_state["chatbot_history"].append({"role": "assistant", "content": error_message})
        # Yield current history with error, reset buttons
        yield ui_state["chatbot_history"], None, None, gr.update(interactive=True), gr.update(interactive=False)
    finally:
        # Ensure thread reference is cleared
        ui_state["agent_thread"] = None


def convert_record_to_steps(record_dir):
    project = os.getenv("LLM_PROJECT")
    location = os.getenv("LLM_LOCATION")

    llm = llm_provider.LLMProvider(
        llm_provider="gemini",
        model=os.getenv("LLM_MODEL", "gemini-2.5-pro"),
        project=project,
        location=location
    )
    agent = RecordToSimpleStepsAgent(llm=llm)
    task_infos = agent.execute(record_dir=record_dir, temperature=0)
    if not task_infos:
        return None, None
    else:
        task_name = task_infos["task_name"]
        task_steps = ""
        for i, action_info in enumerate(task_infos["task_steps"]):
            task_steps += f"{i + 1}.{action_info['action']}\n"
        return task_name, task_steps


def on_stop_click():
    """Handles the 'Stop Agent' button click."""
    global ui_state
    logging.info("Stop Agent button clicked.")
    stopped = False
    if ui_state["agent"]:
        try:
            # Call the agent's stop method - this should signal the agent's internal loop to break
            ui_state["agent"].stop()
            logging.info("Agent stop signal sent.")
            stopped = True
            # Add a message to the chat? (Optional, main loop will update)
            # ui_state["chatbot_history"].append({"role": "assistant", "content": "**Notice:** Stop request sent to agent."})
        except Exception as e:
            logging.error(f"Error trying to stop agent: {e}", exc_info=True)
            # ui_state["chatbot_history"].append({"role": "assistant", "content": f"**Error:** Could not signal agent to stop.\n`{e}`"})

    # Optionally join the thread briefly if it exists, but don't block indefinitely
    if ui_state["agent_thread"] and ui_state["agent_thread"].is_alive():
        logging.info("Waiting briefly for agent thread to stop after signal...")
        ui_state["agent_thread"].join(timeout=1.0)  # Wait max 1 second
        if ui_state["agent_thread"].is_alive():
            logging.warning("Agent thread still alive after stop signal and timeout.")

    # Reset button states: Enable Run, Disable Stop
    return gr.update(interactive=True), gr.update(interactive=False)


def create_ui():
    """Creates the Gradio interface."""

    css = """
    /* Reduce padding/margin around chatbot for potential height gain */
    .header-text {
        text-align: center;
        margin-bottom: 10px; /* Slightly reduced margin */
    }
    /* Reduce spacing below input fields */
    .gradio-Textbox { margin-bottom: 10px !important; }
    """

    scroll_js = """
       function Scrolldown() {
       let targetNode = document.querySelector('[aria-label="chatbot conversation"]')
       // Options for the observer (which mutations to observe)
       const config = { attributes: true, childList: true, subtree: true };

       // Callback function to execute when mutations are observed
       const callback = (mutationList, observer) => {
       setTimeout(() => {
                targetNode.scrollTop = targetNode.scrollHeight;
            }, 100);
       };

       // Create an observer instance linked to the callback function
       const observer = new MutationObserver(callback);

       // Start observing the target node for configured mutations
       observer.observe(targetNode, config);
       }
       """

    with gr.Blocks(title="QAgent-Omni WebUI", css=css, theme=gr.themes.Ocean(),
                   js=scroll_js) as demo:  # Changed theme again

        gr.Markdown(
            """
            # QAgent-Omni WebUI
            ### GUI Automation Agent for Any Platform
            """,
            elem_classes=["header-text"],
        )
        with gr.Row():
            # Left Column: Inputs and Controls
            with gr.Column(scale=3):
                record_dir = gr.Textbox(
                    label="Record Absolute Dir",
                    lines=1,
                    max_lines=1
                )
                extract_btn = gr.Button("Record to Steps", variant="primary")
                task_input = gr.Textbox(
                    label="Task Description",
                    placeholder="Enter the main goal for the agent (e.g., 'Send a message to John saying hello')",
                    lines=3
                )
                task_steps_input = gr.Textbox(
                    label="Task Steps (Optional)",
                    placeholder="Provide specific steps if needed, one per line.",
                    lines=6
                )
                task_infos_input = gr.Textbox(
                    label="Additional Information (Optional)",
                    placeholder="Enter any extra context, like login credentials (use carefully!) or specific data.",
                    lines=3
                )

                with gr.Row():
                    run_btn = gr.Button("Run Agent", variant="primary", scale=2)
                    stop_btn = gr.Button("Stop Agent", variant="stop", interactive=False, scale=1)

            # Middle Column: Agent Step Output (Chatbot)
            with gr.Column(scale=3):  # Give chatbot slightly more relative space
                chatbot_display = gr.Chatbot(
                    label="Agent Steps & Output",
                    show_label=True,
                    autoscroll=False,
                    height=800,
                    type="messages"
                )

            # Right Column: Result GIF/JSON
            with gr.Column(scale=2):
                result_gif = gr.Image(label="Result GIF", type="filepath", interactive=False)
                result_json = gr.File(label="Execution History JSON", interactive=False)

        # Connect the Run button: It now yields multiple updates
        run_btn.click(
            fn=click_to_run_agent,
            inputs=[task_input, task_steps_input, task_infos_input],
            outputs=[chatbot_display, result_gif, result_json, run_btn, stop_btn],
        )

        # Connect the Stop button
        stop_btn.click(
            fn=on_stop_click,
            inputs=[],
            outputs=[run_btn, stop_btn],
        )

        extract_btn.click(
            fn=convert_record_to_steps,
            inputs=[record_dir],
            outputs=[task_input, task_steps_input]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Run QAgent-Omni Gradio WebUI")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind the Gradio server to")
    parser.add_argument("--port", type=int, default=6789, help="Port for the Gradio server")
    parser.add_argument("--share", action='store_true', help="Create a publicly shareable Gradio link")
    args = parser.parse_args()

    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
