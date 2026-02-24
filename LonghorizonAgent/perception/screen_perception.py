import copy
import os
import pdb

import cv2
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import logging
import glob

from .models.icon_detect_model import IconDetectModel
from .models.ocr_model import OCRModel
from ..common import utils

logger = logging.getLogger(__name__)


@dataclass
class PerceptionInfo:
    perception_dim: List[Dict] = field(default_factory=list)
    perception_info: List[Dict] = field(default_factory=list)
    ocr_infos: List[Dict] = field(default_factory=list)
    icon_infos: List[Dict] = field(default_factory=list)
    template_infos: List[Dict] = field(default_factory=list)


class ScreenPerception:
    """
    屏幕感知模块
    """

    def __init__(self, **kwargs):
        if kwargs.get("use_icon_caption", False):
            from .models.icon_caption_model import IconCaptionModel
            self.icon_caption_model = IconCaptionModel()
        else:
            self.icon_caption_model = None
        self.icon_detect_model = IconDetectModel()
        self.ocr_model = OCRModel(use_rec=kwargs.get("use_rec", True))

        icon_template_dir = kwargs.get("icon_template_dir", "")
        self.icon_templates = self.load_icon_templates(icon_template_dir)

    def load_icon_templates(self, icon_template_dir):
        """
        Loads predefined icon templates.

        Args:
            icon_template_dir (str): The directory containing icon templates.  Can also be a glob pattern like "dir1/*,dir2/*".

        Returns:
            dict: A dictionary of icon templates, where the key is the icon name and the value is the icon image.
        """
        logger.info(f"Loading icon templates from path(s): {icon_template_dir}")
        icon_templates = {}

        if icon_template_dir:
            # Handle comma-separated paths for globbing.  Handle a single path as well.
            paths = icon_template_dir.split(",") if "," in icon_template_dir else [icon_template_dir]
            for path in paths:
                path = path.strip()  # Remove leading/trailing whitespace
                if not path:
                    continue  # Skip empty paths

                for tpath in glob.glob(path):  # Use glob to find files and directories
                    if os.path.isdir(tpath):
                        # Skip directories
                        continue

                    # Check if the file is likely an image based on extension. This is a quick and dirty check.
                    ext = os.path.splitext(tpath)[1].lower()
                    if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif']:
                        logger.warning(f"Skipping {tpath}: Not a recognized image file type.")
                        continue

                    try:
                        with open(tpath, 'rb') as f:
                            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                            timg = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

                        if timg is None:
                            logger.warning(f"Skipping {tpath}:  cv2.imdecode failed to decode the image.")
                            continue
                        icon_name = os.path.splitext(os.path.basename(tpath))[
                            0]  # Remove file extension and get filename as key
                        icon_templates[icon_name] = timg
                    except Exception as e:
                        logger.error(f"Failed to load icon template from {tpath}: {e}")

        if icon_templates:
            logger.info(f"Successfully loaded icon templates, total: {len(icon_templates)}")
        else:
            logger.warning("No icon templates provided!")
        return icon_templates

    def _match_templates(self, image_bgr, icon_templates, template_thred=0.7):
        """
        使用模板匹配方法检测图标
        """
        img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        template_infos = []
        for key, template in icon_templates.items():
            try:
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(res >= template_thred)
                if len(locations) == 0 or len(locations[0]) == 0:
                    continue
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
            for loc in zip(*locations[::-1]):
                top_left = loc
                bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

                perception_info = {
                    "type": "icon_match",
                    "text": f"{key}",
                    "box": [int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])],
                }
                template_infos.append(perception_info)
        return template_infos

    def merge_to_get_perception_infos(self, screen_state: PerceptionInfo, **kwargs):
        """
        合并三个信息到统一的icon信息
        :param screen_state:
        :param kwargs:
        :return:
        """
        iou_thred = kwargs.get("iou_thred", 0.7)
        ocr_infos = copy.deepcopy(screen_state.ocr_infos)
        icon_infos = copy.deepcopy(screen_state.icon_infos)
        template_infos = copy.deepcopy(screen_state.template_infos)
        perception_infos = ocr_infos + template_infos

        icon_valid = [[] for _ in range(len(icon_infos))]
        for i, icon_info_ in enumerate(icon_infos):
            for j, pinfo_ in enumerate(perception_infos):
                iou1, iou2 = utils.calculate_iou_v2(icon_info_['box'], pinfo_['box'])
                if max(iou1, iou2) > iou_thred and min(iou1, iou2) > 0.3:
                    if iou1 > iou_thred:
                        icon_valid[i].append(-1)
                        break
                    elif iou2 > iou_thred:
                        icon_valid[i].append(j)

        for i, icon_info_ in enumerate(icon_infos):
            if not icon_valid[i]:
                perception_infos.append(icon_info_)
            elif len(icon_valid[i]) == 1 and icon_valid[i][0] != -1:
                pinfo_ = perception_infos[icon_valid[i][0]]
                pinfo_["box"] = [
                    min(pinfo_["box"][0], icon_info_["box"][0]),
                    min(pinfo_["box"][1], icon_info_["box"][1]),
                    max(pinfo_["box"][2], icon_info_["box"][2]),
                    max(pinfo_["box"][3], icon_info_["box"][3]),
                ]

        return perception_infos

    def caption_icons_wo_text(self, image, screen_state: PerceptionInfo, **kwargs):
        """

        :param screen_state:
        :param kwargs:
        :return:
        """
        if self.icon_caption_model is None:
            return
        if image is None:
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        caption_pinds = []
        for i, pinfo in enumerate(screen_state.perception_info):
            if not pinfo["text"]:
                caption_pinds.append(i)

        caption_icon_bboxs = [screen_state.perception_info[i]["box"] for i in caption_pinds]
        caption_icon_texts = self.icon_caption_model.caption(image_rgb, caption_icon_bboxs)
        for i, j in enumerate(caption_pinds):
            screen_state.perception_info[j]["text"] = caption_icon_texts[i]

    def run_perception(self, image, **kwargs) -> PerceptionInfo:
        screen_state = PerceptionInfo(
            perception_dim=[],
            perception_info=[],
            ocr_infos=[],
            icon_infos=[],
            template_infos=[]
        )
        if image is None:
            return screen_state

        screen_state.perception_dim = [image.shape[1], image.shape[0]]

        img_h, img_w = image.shape[:2]
        split_num_x = kwargs.get("split_x", None)
        if split_num_x is None:
            if img_h > img_w:
                split_num_x = 1
            else:
                split_num_x = 2

        split_num_y = kwargs.get("split_y", None)
        if split_num_y is None:
            if img_h > img_w:
                split_num_y = 2
            else:
                split_num_y = 1

        if kwargs.get("use_ocr", True):
            ocr_results = self.ocr_model.ocr(image,
                                             split_num_x=split_num_x,
                                             split_num_y=split_num_y,
                                             predict_type=kwargs.get("predict_type", "combined"),
                                             split_text=kwargs.get("split_text", False))

            ocr_infos = []
            for ocr_ret in ocr_results:
                text = ocr_ret[1]

                if (text.isalpha() or text.isdigit()) and len(text) <= 1:
                    continue  # Skip this OCR result
                ocr_info_ = {
                    "type": "ocr_text",
                    "box": list(map(int, ocr_ret[0])),
                    "text": ocr_ret[1]
                }

                ocr_infos.append(ocr_info_)
            screen_state.ocr_infos = copy.deepcopy(ocr_infos)

        if kwargs.get("use_icon_detect", True):
            icon_results = self.icon_detect_model.detect(image,
                                                         box_threshold=0.05,
                                                         iou_threshold=0.1,
                                                         imgsz=kwargs.get("detect_imgsz", 640),
                                                         split_num_x=split_num_x,
                                                         split_num_y=split_num_y,
                                                         predict_type=kwargs.get("predict_type", "combined"))
            icon_infos = []
            for icon_ret_ in icon_results:
                icon_info_ = {
                    "type": "icon_detect",
                    "box": list(map(int, icon_ret_[:4])),
                    "text": ""
                }
                icon_infos.append(icon_info_)
            screen_state.icon_infos = copy.deepcopy(icon_infos)

        if kwargs.get("use_template_match", True):
            # template matching
            template_infos = self._match_templates(image, self.icon_templates)
            screen_state.template_infos = copy.deepcopy(template_infos)

        perception_infos = self.merge_to_get_perception_infos(screen_state)
        screen_state.perception_info = perception_infos
        self.caption_icons_wo_text(image, screen_state, **kwargs)
        # sorted the result based on box location
        grid_size = min(img_h, img_w) // kwargs.get("grid_num", 16)
        screen_state.perception_info = sorted(
            screen_state.perception_info,
            key=lambda info: (
                (info["box"][1] + info["box"][3]) / 2 // grid_size * grid_size,
                (info["box"][0] + info["box"][2]) / 2 // grid_size * grid_size)
        )
        return screen_state
