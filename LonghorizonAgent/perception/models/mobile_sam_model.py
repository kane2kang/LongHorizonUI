import pdb

import numpy as np
import torch
import os.path
import cv2
import logging
from huggingface_hub import hf_hub_download
from mobile_sam import sam_model_registry, SamPredictor

from ...common import utils

logger = logging.getLogger(__name__)


class MobileSAMModel:
    """
    MobileSAM Model class for image segmentation.

    This class encapsulates the MobileSAM model and provides image segmentation prediction functionality.
    """

    def __init__(self, **kwargs):
        """
        Initializes the MobileSAMModel class.

        Args:
            **kwargs: Optional parameters, including:
                model_path (str): The path to the model weights file, defaults to constants.CHECKPOINT_DIR/MobileSAM/mobile_sam.pt.
                log_file (str): Path to the log file, defaults to None.
        """
        checkpoint_dir = kwargs.get("checkpoint_dir", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        device, dtype = utils.get_optimize_device_and_dtype()
        model_path = kwargs.get("model_path", os.path.join(checkpoint_dir, "MobileSAM/mobile_sam.pt"))

        # Check if the model file exists, and download it from Hugging Face if it doesn't
        if not os.path.exists(model_path):
            logger.info(f"{model_path} does not exist! Downloading from HuggingFace: dhkim2810/MobileSAM!")
            hf_hub_download(repo_id="dhkim2810/MobileSAM", filename="mobile_sam.pt",
                            local_dir=os.path.join(checkpoint_dir, "MobileSAM"))

        # Load the MobileSAM model
        mobile_sam = sam_model_registry["vit_t"](checkpoint=model_path)
        mobile_sam.to(device=device, dtype=dtype)  # Move the model to the specified device and data type
        mobile_sam.eval()  # Set the model to evaluation mode
        self.predictor = SamPredictor(mobile_sam)  # Create a SAM predictor

    @torch.inference_mode()  # Use inference mode
    def predict(self, image_bgr, point_coords=None, point_labels=None):
        """
        Performs prediction using the MobileSAM model.

        Args:
            image_rgb (np.ndarray): RGB image, shape (H, W, 3).
            point_coords (np.ndarray, optional): Coordinates of clicked points, shape (N, 2).
            point_labels (np.ndarray, optional): Labels of clicked points, shape (N).

        Returns:
             tuple: A tuple containing the segmentation mask and the maximum bounding box.
                mask (np.ndarray): Segmentation mask, shape (H, W), data type np.uint8.
                max_box (list): The maximum bounding box, in the format [x1, y1, x2, y2].
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)  # Set the input image
        masks, _, _ = self.predictor.predict(point_coords, point_labels, multimask_output=False)  # Perform prediction
        mask = (masks[0] * 255).astype(np.uint8)  # Convert the mask to uint8 type
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

        max_area = 0  # Initialize the maximum area
        mask_box = None  # Initialize the maximum bounding box

        # Iterate over all contours
        for contour in contours:
            area = cv2.contourArea(contour)  # Calculate the contour area
            if area > max_area:
                max_area = area  # Update the maximum area
                max_box_ = cv2.boundingRect(contour)  # Get the bounding box
                max_box_ = list(max_box_)  # Convert the bounding box to a list
                max_box_[2] += max_box_[0]  # Calculate the x-coordinate of the bottom-right corner
                max_box_[3] += max_box_[1]  # Calculate the y-coordinate of the bottom-right corner
                mask_box = max_box_  # Update the maximum bounding box
        return mask, mask_box
