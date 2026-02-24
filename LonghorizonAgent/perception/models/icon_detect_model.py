import pdb

import numpy as np
import torch
import logging
import os.path
from huggingface_hub import hf_hub_download
import os

os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO
from ...common import utils

logger = logging.getLogger(__name__)


class IconDetectModel:
    def __init__(self, **kwargs):

        checkpoint_dir = kwargs.get("checkpoint_dir", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = kwargs.get("model_path",
                                os.path.join(checkpoint_dir, "OmniParser-v2.0/icon_detect/model.pt"))
        if not os.path.exists(model_path):
            logger.info(f"{model_path} don't exist! Download from HuggingFce: microsoft/OmniParser-v2.0!")
            hf_hub_download(repo_id="microsoft/OmniParser-v2.0", subfolder="icon_detect", filename="model.pt",
                            local_dir=os.path.join(checkpoint_dir, "OmniParser-v2.0"))
            hf_hub_download(repo_id="microsoft/OmniParser-v2.0", subfolder="icon_detect", filename="model.yaml",
                            local_dir=os.path.join(checkpoint_dir, "OmniParser-v2.0"))

        device, dtype = utils.get_optimize_device_and_dtype()
        self.model = YOLO(model_path, verbose=False).to(device=device, dtype=dtype) # 强制所有参数为 float32
        for param in self.model.model.parameters():
            param.requires_grad = False
            param.data = param.data.to(dtype=torch.float32)#

    @torch.inference_mode()
    def detect(self, image, **kwargs):
        """
        使用 YOLO 模型进行图标检测。

        Args:
            image (np.ndarray): 输入图像，形状为 (H, W, 3)。BGR 格式
            **kwargs: 可选参数，包括:
                split_infer (bool): 是否进行分块推理，默认为 True。
                box_threshold (float): 检测框置信度阈值，默认为 0.05。
                iou_threshold (float): IOU 阈值，用于 NMS，默认为 0.4。
                imgsz (int): 输入图像大小，默认为 1280。
                split_num_x (int):  图像在 x 轴上分割的数量
                split_num_y (int):  图像在 y 轴上分割的数量

        Returns:
            List[List[float]]: 检测到的图标边界框列表，每个边界框格式为 [x1, y1, x2, y2, confidence]。
        """
        h, w = image.shape[:2]
        predict_type = kwargs.get("predict_type", "single")

        if predict_type == "split":
            icon_bboxs = self.predict_split(image, w, h, **kwargs)
        elif predict_type == "single":
            icon_bboxs = self.predict_single(image, w, h, **kwargs)
        elif predict_type == "combined":
            icon_bboxs1 = self.predict_single(image, w, h, **kwargs)
            icon_bboxs2 = self.predict_split(image, w, h, **kwargs)
            icon_bboxs = utils.merge_icon_bboxs_v2(icon_bboxs1, icon_bboxs2, iou_threshold=0.4)
        else:
            raise NotImplemented
        return icon_bboxs

    def predict_split(self, image, w, h, **kwargs):
        """
        将图像分割成多个小块并分别进行预测。

        Args:
            image (np.ndarray): 输入图像。
            w (int): 图像宽度。
            h (int): 图像高度。
            **kwargs:  参数传递

        Returns:
            List[List[float]]: 所有小块中检测到的边界框列表。
        """
        split_num_x = kwargs.get("split_num_x", 1)
        split_num_y = kwargs.get("split_num_y", 1)

        x_splits = np.linspace(0, w, split_num_x + 1, dtype=int).tolist()
        y_splits = np.linspace(0, h, split_num_y + 1, dtype=int).tolist()

        all_icon_bboxs = []
        for i in range(split_num_y):
            for j in range(split_num_x):
                x_start, x_end = x_splits[j], x_splits[j + 1]
                y_start, y_end = y_splits[i], y_splits[i + 1]

                crop_img = image[y_start:y_end, x_start:x_end]
                crop_icon_bboxs = self._predict_split(crop_img, **kwargs)

                # 调整坐标系到原图
                if len(crop_icon_bboxs) > 0:
                    crop_icon_bboxs[:, [0, 2]] += x_start
                    crop_icon_bboxs[:, [1, 3]] += y_start
                    all_icon_bboxs.append(crop_icon_bboxs)

        # 合并分块结果
        if all_icon_bboxs:
            icon_bboxs = np.concatenate(all_icon_bboxs, axis=0)

            # 修正分割线附近的框
            icon_bboxs = self._merge_split_results(icon_bboxs, x_splits, y_splits, w, h, **kwargs)
        else:
            # icon_bboxs = [] # raw
            icon_bboxs = np.array([])  # 确保初始化为 numpy 数组

        return icon_bboxs.tolist()  # 返回列表形式

    def _predict_split(self, image, **kwargs) -> np.ndarray:
        """
        在单个图像上进行预测

        Args:
            image (np.ndarray): 输入图像
            **kwargs: 其余的参数

        Returns:
            np.ndarray: 检测到的边界框
        """
        result = self.model.predict(
            source=image,
            conf=kwargs.get("box_threshold", 0.05),
            iou=kwargs.get("iou_threshold", 0.4),
            imgsz=kwargs.get("imgsz", 1280),
        )
        icon_bboxs = result[0].boxes.xyxy.cpu().numpy()
        if len(icon_bboxs):
            scores = result[0].boxes.conf.cpu().numpy()
            icon_bboxs = np.concatenate([icon_bboxs, scores[:, None]], axis=-1)
            icon_bboxs = utils.remove_boxes(icon_bboxs.tolist(), (image.shape[1], image.shape[0]),
                                            iou_threshold=kwargs.get("iou_thred", 0.5))

        return np.array(icon_bboxs)

    def predict_single(self, image, w, h, **kwargs):
        """
        在完整图像上进行预测
        """
        result = self.model.predict(
            source=image,
            conf=kwargs.get("box_threshold", 0.05),
            iou=kwargs.get("iou_threshold", 0.4),
            imgsz=kwargs.get("imgsz", (h, w)),
        )
        icon_bboxs = result[0].boxes.xyxy.cpu().numpy()
        if len(icon_bboxs):
            scores = result[0].boxes.conf.cpu().numpy()
            icon_bboxs = np.concatenate([icon_bboxs, scores[:, None]], axis=-1)
            icon_bboxs = utils.remove_boxes(icon_bboxs.tolist(), (w, h), iou_threshold=kwargs.get("iou_thred", 0.5))
        else:
            icon_bboxs = []
        return icon_bboxs

    def _merge_split_results(self, icon_bboxs, x_splits, y_splits, w, h, **kwargs):
        """
        合并分块推理的结果，并处理靠近分割线的边界框。

        Args:
            icon_bboxs (np.ndarray): 所有分块的边界框。
            x_splits (np.ndarray): X轴分割线位置。
            y_splits (np.ndarray): Y轴分割线位置。
            w (int): 原始图像的宽度。
            h (int): 原始图像的高度。
            **kwargs: 参数传递

        Returns:
            np.ndarray: 合并后的边界框列表。
        """

        line_threshold = 10  # 分割线附近的阈值

        # 分别处理靠近垂直和水平分割线的文本框
        icon_bboxs_x_line = []
        icon_bboxs_y_line = []
        icon_bboxs_rest = []

        for box_ in icon_bboxs:
            is_x_line_box = False
            is_y_line_box = False

            # 检查是否靠近垂直分割线
            for x_split in x_splits[1:-1]:  # 不包括首尾
                if abs(box_[0] - x_split) < line_threshold or abs(box_[2] - x_split) < line_threshold:
                    is_x_line_box = True
                    break

            # 检查是否靠近水平分割线
            for y_split in y_splits[1:-1]:  # 不包括首尾
                if abs(box_[1] - y_split) < line_threshold or abs(box_[3] - y_split) < line_threshold:
                    is_y_line_box = True
                    break

            if is_x_line_box:
                icon_bboxs_x_line.append(box_)
            elif is_y_line_box:
                icon_bboxs_y_line.append(box_)
            else:
                icon_bboxs_rest.append(box_)

        if icon_bboxs_rest:
            icon_bboxs = np.array(icon_bboxs_rest)
        else:
            icon_bboxs = np.empty((0, 5), dtype=np.float32)
        if icon_bboxs_x_line:
            # 合并靠近垂直分割线的文本框
            merged_results_x = self._merge_line_boxes(icon_bboxs_x_line)
            icon_bboxs = np.concatenate([icon_bboxs, np.array(merged_results_x)], axis=0)

        if icon_bboxs_y_line:
            # 合并靠近水平分割线的文本框
            merged_results_y = self._merge_line_boxes(icon_bboxs_y_line)
            icon_bboxs = np.concatenate([icon_bboxs, np.array(merged_results_y)], axis=0)

        return icon_bboxs

    def _merge_line_boxes(self, icon_bboxs_line):
        """
        合并靠近分割线的边界框。

        Args:
            icon_bboxs_line (List[List[float]]): 靠近分割线的边界框列表。

        Returns:
            List[List[float]]: 合并后的边界框列表。
        """
        icon_bboxs_merge = []
        for box_1 in icon_bboxs_line:
            flag = True
            for box_2 in icon_bboxs_merge:
                if min(box_1[2], box_2[2]) > max(box_1[0], box_2[0]) or min(box_1[3], box_2[3]) > max(box_1[1],
                                                                                                      box_2[1]):
                    box_2[0] = min(box_1[0], box_2[0])
                    box_2[1] = min(box_1[1], box_2[1])
                    box_2[2] = max(box_1[2], box_2[2])
                    box_2[3] = max(box_1[3], box_2[3])
                    flag = False
                    break
            if flag:
                icon_bboxs_merge.append(box_1)
        return icon_bboxs_merge
