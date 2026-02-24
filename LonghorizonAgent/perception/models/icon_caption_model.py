import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from torchvision.transforms import ToPILImage
from huggingface_hub import snapshot_download
from tqdm import tqdm
import logging
import os

from ...common import utils

logger = logging.getLogger(__name__)


class IconCaptionModel:
    def __init__(self, **kwargs):
        self.device, self.dtype = utils.get_optimize_device_and_dtype()
        checkpoint_dir = kwargs.get("checkpoint_dir", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True,
                                                       local_dir=checkpoint_dir)
        model_path = kwargs.get("model_path",
                                os.path.join(checkpoint_dir, "OmniParser-v2.0/icon_caption/model.safetensors"))
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            logger.info(f"Icon caption model does not exist! Downloading from HuggingFace: microsoft/OmniParser!")
            snapshot_download(repo_id="microsoft/OmniParser-v2.0", allow_patterns="icon_caption/*",
                              local_dir=os.path.join(checkpoint_dir, "OmniParser-v2.0"))



        # Check if the flash-attn package is installed
        use_flash_attention = False
        try:
            import importlib
            # Check if the flash-attn package is installed
            if importlib.util.find_spec("flash_attn"):
                use_flash_attention = True
                logger.info("flash-attn is installed, will use flash_attention_2")
            else:
                logger.info("flash-attn is not installed, will not use flash_attention_2")
        except ImportError:
            logger.warning("flash-attn is not installed, will not use flash_attention_2")

        attn_kwargs = {}
        if use_flash_attention:
            attn_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(os.path.dirname(model_path),
                                                          torch_dtype=self.dtype,
                                                          device_map=self.device,
                                                          local_files_only=False,
                                                          trust_remote_code=True,
                                                          **attn_kwargs
                                                          ).eval()
        self.to_pil = ToPILImage()

    @torch.inference_mode()
    def caption(self, image, icon_bboxs, **kwargs):
        cropped_pil_images = []
        # Crop icon regions and convert to PIL images
        for i, coord in enumerate(icon_bboxs):
            xmin, xmax = int(coord[0]), int(coord[2])
            ymin, ymax = int(coord[1]), int(coord[3])
            cropped_image = image[ymin:ymax, xmin:xmax, :]
            cropped_pil_images.append(self.to_pil(cropped_image))

        prompt = kwargs.get("prompt", None)
        if not prompt:
            prompt = "<CAPTION>"  # Default prompt

        batch_size = kwargs.get("infer_bs", 16)  # Number of samples per batch
        generated_texts = []  # Store generated caption texts

        # Process cropped icons in batches
        for i in tqdm(range(0, len(cropped_pil_images), batch_size)):
            batch = cropped_pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(
                device=self.device,
                dtype=self.dtype)  # Preprocess input images and prompts

            # Generate caption texts
            generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                                max_new_tokens=20, num_beams=1, do_sample=False)
            generated_text = self.processor.batch_decode(generated_ids,
                                                         skip_special_tokens=True)  # Decode generated texts
            generated_text = [gen.strip() for gen in generated_text]  # Remove leading/trailing whitespace
            generated_texts.extend(generated_text)  # Add generated texts to the list
        return generated_texts
