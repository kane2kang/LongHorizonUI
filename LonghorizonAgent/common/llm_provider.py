import copy
import base64
import pdb

from openai import AzureOpenAI, OpenAI, OpenAIError
from google import genai
from google.genai import types as genai_types  # Use alias to avoid conflict
from google.api_core import exceptions as google_exceptions
import os
import json
import requests
from PIL import Image
import io
from typing import List, Dict, Any, Tuple, Optional
from google.oauth2 import service_account
import logging
from typing import List, Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)


def _fetch_image_data(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Fetches image data and determines MIME type from URL or base64 data URI."""
    if url.startswith("data:image"):
        # Handle base64 data URI (e.g., data:image/png;base64,iVBORw0KGgo...)
        try:
            header, encoded = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]  # Extract mime_type
            data = base64.b64decode(encoded)
            return data, mime_type
        except Exception as e:
            logger.error(f"Error decoding base64 image URI: {e}", exc_info=True)
            return None, None
    else:
        try:
            response = requests.get(url, stream=True, timeout=20)  # Added timeout
            response.raise_for_status()  # Check for HTTP errors
            content_type = response.headers.get("Content-Type")
            mime_type = None
            if content_type:
                mime_type = content_type.split(';')[0].strip().lower()  # Clean mime type

            img_data = response.content
            return img_data, mime_type
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching image from URL '{url}': {e}", exc_info=True)
            return None, None
        except Exception as e:
            logger.error(f"Unknown error processing image from URL '{url}': {e}", exc_info=True)
            return None, None


def _create_image_part(url: str) -> Optional[genai_types.Part]:
    """
    Creates a Gemini Image Part from a URL. Tries to handle/convert unsupported formats.
    """
    SUPPORTED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}

    image_data, mime_type = _fetch_image_data(url)

    if not image_data:
        logger.warning(f"Could not retrieve image data for URL: {url}")
        return None

    # Use Pillow to validate, get a reliable MIME type, and convert if necessary
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            pil_format = img.format
            if not pil_format:
                raise ValueError("Pillow could not identify image format")

            detected_mime_type = f"image/{pil_format.lower()}"
            mime_type = detected_mime_type  # Prefer Pillow's detected type

            if mime_type not in SUPPORTED_MIME_TYPES:
                logger.warning(
                    f"Image format '{mime_type}' not directly supported. Attempting conversion to PNG. URL: {url}")
                # Attempt conversion to a supported format (e.g., PNG)
                output_format = "PNG"
                output_mime = "image/png"
                if output_mime not in SUPPORTED_MIME_TYPES:
                    logger.error(f"Target conversion format {output_mime} is also not supported!")
                    return None

                img_byte_arr = io.BytesIO()
                # Save transparency if original had it, otherwise convert to RGB first for broader compatibility
                if img.mode in ('RGBA', 'LA', 'P'):
                    img.save(img_byte_arr, format=output_format)
                else:
                    rgb_img = img.convert("RGB")
                    rgb_img.save(img_byte_arr, format=output_format)

                image_data = img_byte_arr.getvalue()
                mime_type = output_mime  # Update to the new type
                logger.info(f"Image successfully converted to {mime_type}. URL: {url}")

            # Create Gemini Blob
            image_part_data = genai_types.Part.from_bytes(
                mime_type=mime_type,
                data=image_data)
            return image_part_data

    except Exception as e:
        # Handle Pillow errors (open, conversion, unsupported format)
        logger.error(f"Error processing image with Pillow ({mime_type=}): {e}. URL: {url}", exc_info=True)
        return None


class LLMProvider:
    """
    A unified interface for interacting with different LLM providers.
    """
    SUPPORTED_PROVIDERS = Literal["openai", "azure_openai", "gemini"]

    def __init__(self,
                 llm_provider: SUPPORTED_PROVIDERS,
                 model: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,  # Required for Azure/Custom OpenAI
                 # Gemini specific args
                 project: Optional[str] = None,
                 location: Optional[str] = None,
                 google_key_json_path: Optional[str] = None,
                 # Azure specific args
                 api_version: str = "2025-01-01-preview",
                 **kwargs):
        """
        Initializes the LLM provider client.

        Args:
            llm_provider: The name of the provider ('openai', 'azure', 'gemini').
            model: The specific model name to use.
            api_key: API key for the provider (OpenAI, Azure). Can often be set via env vars too.
            base_url: Base endpoint URL. Required for Azure and custom OpenAI deployments.
            project: Google Cloud Project ID (Required for Gemini Vertex AI).
            location: Google Cloud Location/Region (Required for Gemini Vertex AI).
            google_key_json_path: Path to the Google service account JSON key file (Optional, uses ADC if not provided).
            api_version: Azure API version (default: "2025-01-01-preview").
        """
        self.llm_provider = llm_provider.lower()
        self.model = model
        self.client = None
        self._validate_config(llm_provider, api_key, base_url, project, location)

        logger.info(f"Initializing LLM provider: {self.llm_provider} with model: {self.model}")

        try:
            if self.llm_provider == "openai":
                # Key can also be read from OPENAI_API_KEY env var by the library
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            elif self.llm_provider == "azure_openai":
                # Key can also be read from AZURE_OPENAI_API_KEY env var
                self.client = AzureOpenAI(
                    azure_endpoint=base_url,
                    api_key=api_key,
                    api_version=api_version
                )
            elif self.llm_provider == "gemini":
                self.client = self._init_gemini_client(google_key_json_path, project, location)
            else:
                raise ValueError(
                    f"Unsupported LLM provider: {self.llm_provider}. Supported: {self.SUPPORTED_PROVIDERS}")

            logger.info(f"{self.llm_provider.capitalize()} client initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize {self.llm_provider.capitalize()} client: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize {self.llm_provider.capitalize()} client.") from e

    def _validate_config(self, provider, api_key, base_url, project, location):
        """Basic validation of required arguments per provider."""
        if provider == "azure_openai":
            if not base_url:
                raise ValueError("Azure provider requires 'base_url' (azure_endpoint).")
            # API key might be optional if set via env var, but usually good practice to require explicit pass or env var check
            if not api_key and not os.getenv("AZURE_OPENAI_API_KEY"):
                logger.warning("Azure API key not provided directly or via AZURE_OPENAI_API_KEY env var.")
                # Decide if you want to raise ValueError here or let the Azure client handle it.
                # raise ValueError("Azure provider requires 'api_key' or AZURE_OPENAI_API_KEY env var.")
        elif provider == "gemini":
            # Using Vertex AI client
            if not project:
                raise ValueError("Gemini (Vertex AI) provider requires 'project' ID.")
            if not location:
                raise ValueError("Gemini (Vertex AI) provider requires 'location'.")
        elif provider == "openai":
            # API key might be optional if set via env var
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                logger.warning("OpenAI API key not provided directly or via OPENAI_API_KEY env var.")

    def _init_gemini_client(self, google_key_json_path, project, location):
        """Initializes the Google Generative AI client, preferring Vertex AI."""
        credentials = None
        effective_key_path = google_key_json_path or os.getenv("GOOGLE_API_KEY_JSON")

        if effective_key_path and os.path.exists(effective_key_path):
            try:
                if effective_key_path.endswith(".json"):
                    with open(effective_key_path, "r") as fin:
                        key_info = json.load(fin)
                        credentials = service_account.Credentials.from_service_account_info(key_info, scopes=[
                            "https://www.googleapis.com/auth/cloud-platform"])
                else:
                    from .utils import load_json_from_encrypted_file
                    key_info = load_json_from_encrypted_file(effective_key_path)
                    credentials = service_account.Credentials.from_service_account_info(key_info, scopes=[
                        "https://www.googleapis.com/auth/cloud-platform"])
            except Exception as e:
                logger.error(f"Error loading credentials from {effective_key_path}: {e}", exc_info=True)
                # Decide whether to raise or fall back to ADC

        if credentials:
            return genai.Client(vertexai=True, project=project, location=location, credentials=credentials)
        else:
            logger.info(
                "No valid service account credentials found or provided, attempting to use Application Default Credentials (ADC) for Vertex AI.")
            # ADC will be used automatically by the client if credentials=None
            return genai.Client(vertexai=True, project=project, location=location)

    def _convert_to_gemini_messages(self, chat_messages: List[Dict[str, Any]]) -> Tuple[
        List[genai_types.Content], Optional[str]]:
        """Converts OpenAI-style message list to Gemini's Content list."""
        gemini_contents: List[genai_types.Content] = []
        system_prompts_list: List[str] = []

        for msg in chat_messages:
            role = msg.get("role")
            content = msg.get("content")

            # 1. Handle system messages -> extract to system_prompts_list
            if role == "system":
                system_prompts_list.append(content[0]["text"])
                continue  # Skip adding system messages to gemini_contents

            # 2. Map roles (only user/assistant are relevant for Gemini history)
            if role == "assistant":
                mapped_role = "model"
            elif role == "user":
                mapped_role = "user"
            else:
                logger.warning(f"Unsupported role '{role}', skipping message: {msg}")
                continue

            # 3. Process content
            if not content:
                logger.warning(f"Empty content for role '{role}', skipping message: {msg}")
                continue

            data_parts: List[genai_types.Part] = []

            # a. Content is a string (text-only)
            if isinstance(content, str):
                if content.strip():
                    data_parts.append(genai_types.Part.from_text(text=content))
                else:
                    logger.warning(f"Text content for role '{role}' is empty or whitespace, skipping.")
                    continue

            # b. Content is a list (multi-modal: text + images)
            elif isinstance(content, list):
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            data_parts.append(genai_types.Part.from_text(text=text))
                        else:
                            logger.warning("Empty or invalid text part in multi-part message ignored.")
                    elif item_type == "image_url":
                        image_url_data = item.get("image_url")
                        if isinstance(image_url_data, dict):
                            url = image_url_data.get("url")
                            if isinstance(url, str) and url.strip():
                                # Process the image URL
                                image_part = _create_image_part(url)
                                if image_part:
                                    data_parts.append(image_part)
                                else:
                                    logger.warning(f"Could not process or skipping image URL: {url}")
                                    # Note: Failed images are skipped, message continues with other parts if any.
                            else:
                                logger.warning("image_url dictionary's 'url' is empty or invalid.")
                        else:
                            logger.warning("Invalid image_url format (expected dictionary).")
                    else:
                        logger.warning(f"Unsupported type in multi-part content: '{item_type}'.")

                if not data_parts:  # If list processing yields no valid parts
                    logger.warning(f"List content for role '{role}' resulted in no valid parts, skipping message.")
                    continue

            # c. Unexpected content type
            else:
                logger.warning(
                    f"Unsupported content type '{type(content)}' for role '{role}'. Attempting str()."
                )
                try:
                    str_content = str(content)
                    if str_content.strip():
                        data_parts.append(genai_types.Part.from_text(text=str_content))
                    else:
                        logger.warning("String conversion resulted in empty content, skipping.")
                        continue
                except Exception as e:
                    logger.error(f"Failed to convert content to string: {e}", exc_info=True)
                    continue  # Skip unprocessable message

            # 4. Create Gemini Content object if parts were generated
            if data_parts:
                # Optional: Add checks for alternating user/model roles if needed, though SDK might handle it.
                gemini_contents.append(genai_types.Content(role=mapped_role, parts=data_parts))

        # 5. Combine system prompts
        final_system_prompt = "\n\n".join(system_prompts_list) if system_prompts_list else None
        return gemini_contents, final_system_prompt

    def add_message(self, role, prompt, chat_history, images=None):
        """
        Add message to chat_history.
        :param role:
        :param prompt:
        :param chat_history:
        :param images:
        :return:
        """
        if images is None:
            images = []
        content = [
            {
                "type": "text",
                "text": prompt
            },
        ]
        if images:
            for base64_image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                )
        chat_history.append({"role": role, "content": content})
        return chat_history

    def invoke(self,
               chat_messages: List[Dict[str, Any]],
               temperature: float = 0.5,
               max_tokens: Optional[int] = None,
               **kwargs) -> Optional[str]:
        """
        Invokes the configured LLM with the given chat messages and parameters.

        Args:
            chat_messages: A list of message dictionaries, expected in OpenAI format:
                           [{"role": "user/assistant/system", "content": "text" or list}, ...]
                           Note: For Gemini, system messages are handled via kwargs['system_prompt'].
            temperature: The sampling temperature (creativity).
            max_tokens: The maximum number of tokens to generate. Note mapping for different providers.
            **kwargs: Additional provider-specific parameters.
                      - OpenAI/Azure: 'seed', etc.
                      - Gemini: 'system_prompt' (str), 'top_p', 'top_k', 'safety_settings' (List[SafetySetting]), etc.

        Returns:
            The text content of the LLM's response, or None if an error occurred.
        """
        logger.info(f"Invoking {self.llm_provider} model '{self.model}' with {len(chat_messages)} messages.")
        if not self.client:
            logger.error("LLM client is not initialized.")
            return None

        try:
            if self.llm_provider in ["openai", "azure_openai"]:
                if not chat_messages:
                    logger.error("No valid messages found for OpenAI/Azure API call.")
                    return None

                max_tokens = max_tokens if max_tokens is not None else 2048
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "messages": chat_messages,
                    "temperature": temperature,
                    **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                    **{k: v for k, v in kwargs.items() if
                       k in ["seed", "top_p", "frequency_penalty", "presence_penalty"]}
                }

                response = self.client.chat.completions.create(**api_params)
                content = response.choices[0].message.content
                return content.strip() if content else None

            elif self.llm_provider == "gemini":
                # Convert messages from OpenAI format to Gemini format
                gemini_chat_messages, system_prompt = self._convert_to_gemini_messages(chat_messages)
                if not gemini_chat_messages:
                    logger.error("No valid messages found for Gemini API call after conversion.")
                    return None

                gemini_max_tokens = max_tokens if max_tokens is not None else 8192
                if not system_prompt:
                    system_prompt = "You are a helpful AI operating assistant, focus on mobile-use, computer-use and browser-use. You need to help me operate the system to complete the user\'s instruction."
                generate_content_config = genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=gemini_max_tokens,
                    response_modalities=["TEXT"],
                    safety_settings=[genai_types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ), genai_types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ), genai_types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ), genai_types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )],
                    system_instruction=[genai_types.Part.from_text(text=system_prompt)]
                )

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=gemini_chat_messages,
                    config=generate_content_config,
                )

                # Extract text content - handle potential errors or empty parts
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    content = response.candidates[0].content.parts[0].text
                    logger.info(f"Received response from {self.llm_provider}.")
                    # logger.debug(f"Response content: {content[:100]}...") # Optional debug
                    return content.strip() if content else None
                else:
                    # Log finish reason if available
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                    safety_ratings = response.candidates[0].safety_ratings if response.candidates else "N/A"
                    logger.warning(
                        f"Gemini response was empty or incomplete. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}")
                    # Check prompt feedback too
                    if hasattr(response, 'prompt_feedback'):
                        logger.warning(f"Gemini prompt feedback: {response.prompt_feedback}")
                    return None

        except (OpenAIError, google_exceptions.GoogleAPIError) as api_err:
            logger.error(f"API error during {self.llm_provider} invocation: {api_err}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during {self.llm_provider} invocation: {e}", exc_info=True)
            return None

        return None
