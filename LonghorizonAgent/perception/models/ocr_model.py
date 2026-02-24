import pdb

import cv2
import os
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import logging
import re
import string

from ...common import utils

logger = logging.getLogger(__name__)


class OCRModel:
    def __init__(self, **kwargs):
        self.engine = RapidOCR(**kwargs)

    def ocr(self, image, **kwargs):
        """
        OCR检测，支持single, split 和 combined 三种模式。

        Args:
            image (np.ndarray): 输入图像 (BGR 格式)。
            **kwargs:
                predict_type (str): "single", "split", 或 "combined"。默认为 "single"。
                split_num_x (int): 图像在 x 轴上分割的数量，仅在 predict_type="split" 时有效。默认为 1。
                split_num_y (int): 图像在 x 轴上分割的数量，仅在 predict_type="split" 时有效。默认为 1。
                iou_threshold (float): IOU 阈值，用于 NMS，默认为 0.4。

        Returns:
            list: OCR结果列表，每个元素包含 (box, text, score)。
        """
        h, w = image.shape[:2]
        predict_type = kwargs.get("predict_type", "single")

        if predict_type == "split":
            ocr_results = self.ocr_split(image, w, h, **kwargs)
        elif predict_type == "single":
            ocr_results = self.ocr_single(image, w, h, **kwargs)
        elif predict_type == "combined":
            ocr_results1 = self.ocr_single(image, w, h, **kwargs)
            ocr_results2 = self.ocr_split(image, w, h, **kwargs)
            ocr_results = self.merge_ocr_results(ocr_results1, ocr_results2,
                                                 iou_threshold=kwargs.get("iou_threshold", 0.4))
        else:
            raise ValueError(f"Invalid predict_type: {predict_type}")
        if kwargs.get("split_text", False):
            ocr_results = self._split_text(ocr_results)
        return ocr_results

    def ocr_single(self, image, w, h, **kwargs):
        """在完整图像上进行 OCR 检测。"""
        result, _ = self.engine(image, use_cls=False)
        return self._format_ocr_result(result, w, h)

    def ocr_split(self, image, w, h, **kwargs):
        """分割图像进行 OCR 检测."""
        split_num_x = kwargs.get("split_num_x", 1)
        split_num_y = kwargs.get("split_num_y", 1)

        x_splits = np.linspace(0, w, split_num_x + 1, dtype=int).tolist()
        y_splits = np.linspace(0, h, split_num_y + 1, dtype=int).tolist()

        all_ocr_results = []
        for i in range(split_num_y):
            for j in range(split_num_x):
                x_start, x_end = x_splits[j], x_splits[j + 1]
                y_start, y_end = y_splits[i], y_splits[i + 1]

                crop_img = image[y_start:y_end, x_start:x_end]
                crop_result, _ = self.engine(crop_img, use_cls=False)
                crop_ocr_results = self._format_ocr_result(crop_result, w, h)

                # 调整坐标系
                for box, text, score in crop_ocr_results:
                    x1, y1, x2, y2 = box
                    x1 += x_start
                    y1 += y_start
                    x2 += x_start
                    y2 += y_start
                    all_ocr_results.append(([x1, y1, x2, y2], text, score))

        # 合并分块结果
        if all_ocr_results:
            all_ocr_results = self._merge_split_ocr_results(all_ocr_results, x_splits, y_splits, w, h, **kwargs)

        return all_ocr_results

    def _format_ocr_result(self, result, w, h):
        """将RapidOCR的result转换成标准格式."""
        formatted_results = []
        if result is not None:  # 加上判断条件
            for ret in result:
                if len(ret) == 3:
                    box, text, score = ret
                else:
                    box = ret
                    text = ""
                    score = 1.0
                x1, y1 = box[0]
                x2, y2 = box[2]
                # remove small text
                if x2 - x1 < w / 40 and y2 - y1 < h / 40:
                    continue
                # add some padding to box
                padding_len = min(y2 - y1, x2 - x1) // 4
                x1 = max(0, x1 - padding_len)
                y1 = max(0, y1 - padding_len)
                x2 = max(0, x2 + padding_len)
                y2 = max(0, y2 + padding_len)
                formatted_results.append(([int(x1), int(y1), int(x2), int(y2)], text, score))
        return formatted_results

    def _merge_split_ocr_results(self, ocr_results, x_splits, y_splits, w, h, **kwargs):
        """
        合并分块推理的 OCR 结果，并处理靠近分割线的文本框。

        Args:
            ocr_results (list): 所有分块的 OCR 结果，每个元素是 (box, text, score)。
            x_splits (np.ndarray): X轴分割线位置。
            y_splits (np.ndarray): Y轴分割线位置。
            w (int): 原始图像的宽度。
            h (int): 原始图像的高度。
            **kwargs: 参数传递

        Returns:
            list: 合并后的 OCR 结果列表。
        """

        line_threshold = kwargs.get("line_threshold", 10)  # 分割线附近的阈值

        # 分别处理靠近垂直和水平分割线的文本框
        ocr_results_x_line = []
        ocr_results_y_line = []
        ocr_results_rest = []

        for box, text, score in ocr_results:
            x1, y1, x2, y2 = box
            is_x_line_box = False
            is_y_line_box = False

            for x_split in x_splits[1:-1]:  # 不包括首尾
                if abs(x1 - x_split) < line_threshold or abs(x2 - x_split) < line_threshold:
                    is_x_line_box = True
                    break

            for y_split in y_splits[1:-1]:  # 不包括首尾
                if abs(y1 - y_split) < line_threshold or abs(y2 - y_split) < line_threshold:
                    is_y_line_box = True
                    break

            if is_x_line_box:
                ocr_results_x_line.append((box, text, score))
            elif is_y_line_box:
                ocr_results_y_line.append((box, text, score))
            else:
                ocr_results_rest.append((box, text, score))

        if ocr_results_x_line:
            merged_results_x = self._merge_line_ocr_boxes(ocr_results_x_line, is_horizontal=False)
            ocr_results_rest += merged_results_x
        if ocr_results_y_line:
            merged_results_y = self._merge_line_ocr_boxes(ocr_results_y_line, is_horizontal=True)
            ocr_results_rest += merged_results_y

        return ocr_results_rest

    def _merge_line_ocr_boxes(self, ocr_results_line, is_horizontal=False):
        """
        合并靠近分割线的文本框，并保证文本顺序正确。

        Args:
            ocr_results_line (list): 靠近分割线的 OCR 结果列表。
            is_horizontal (bool): 是否为水平方向的分割线。
        """

        # 对ocr_results_line根据左上角的x或y坐标进行排序
        if not is_horizontal:
            ocr_results_line = sorted(ocr_results_line, key=lambda x: x[0][0])  # 按照x1排序
        else:
            ocr_results_line = sorted(ocr_results_line, key=lambda x: x[0][1])  # 按照y1排序

        ocr_results_merge = []
        skip = [False] * len(ocr_results_line)  # 用于标记已经合并的框
        for i in range(len(ocr_results_line)):
            if skip[i]:
                continue
            box_1, text_1, score_1 = ocr_results_line[i]
            flag = True
            for j in range(i + 1, len(ocr_results_line)):  # Need index for modification
                if skip[j]:
                    continue
                box_2, text_2, score_2 = ocr_results_line[j]

                x1_1, y1_1, x2_1, y2_1 = box_1
                x1_2, y1_2, x2_2, y2_2 = box_2
                if min(x2_1, x2_2) > max(x1_1, x1_2) or min(y2_1, y2_2) > max(y1_1, y1_2):
                    # 进行合并
                    new_x1 = min(x1_1, x1_2)
                    new_y1 = min(y1_1, y1_2)
                    new_x2 = max(x2_1, x2_2)
                    new_y2 = max(y2_1, y2_2)
                    new_box = [new_x1, new_y1, new_x2, new_y2]

                    # 合并文本
                    new_text = text_1 + text_2

                    new_score = (score_1 + score_2) / 2  # 或者其他合并score的方式

                    ocr_results_merge.append((new_box, new_text, new_score))
                    skip[i] = skip[j] = True
                    flag = False
                    break

            if flag:
                ocr_results_merge.append((box_1, text_1, score_1))
        return ocr_results_merge

    def merge_ocr_results(self, results1, results2, iou_threshold=0.4):
        """
        合并两个 OCR 结果列表。  简单进行拼接
        """
        merged_results = []

        # 将 results1 中的所有文本框添加到 merged_results 中
        merged_results.extend(results1)

        for box2, text2, score2 in results2:
            # 初始化一个标志，用于表示是否需要将 box2 添加到 merged_results 中
            add_box2 = True

            for i, (box1, text1, score1) in enumerate(results1):
                # 计算 box1 和 box2 的 IOU
                iou1, iou2 = utils.calculate_iou_v2(box1, box2)
                iou = max(iou1, iou2)

                if iou > iou_threshold:
                    add_box2 = False
                    break

            if add_box2:
                merged_results.append((box2, text2, score2))

        return merged_results

    def _split_text(self, ocr_results):
        """
        对 OCR 结果的文本进行拆分，根据空格数量、英文单词大小写、中文标点符号等规则。

        Args:
            ocr_results (list): OCR 结果列表，每个元素是 (box, text, score)。 box格式为[x1, y1, x2, y2]

        Returns:
            list: 拆分后的 OCR 结果列表。
        """
        split_ocr_results = ocr_results  # 初始化为原始结果，迭代处理

        new_split_ocr_results = []
        for box, text, score in split_ocr_results:
            if re.search(r"[a-zA-Z]", text):  # 判断是否包含英文字母
                words = text.split()
                new_words = []
                start = 0
                for i in range(1, len(words)):
                    if words[i].strip()[0].isupper() and (
                            words[i - 1].strip()[0].isupper() or words[i - 1].strip()[-1] in ['。', '.', '！', '？', '?',
                                                                                              '!']):
                        new_words.append(" ".join(words[start:i]))
                        start = i
                new_words.append(" ".join(words[start:]))

                if len(new_words) > 1:  # 说明进行了拆分
                    total_len = len(text)
                    len_ratios = [len(s) / total_len for s in new_words]
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    current_x = x1
                    for i, split_text in enumerate(new_words):
                        new_x2 = current_x + width * len_ratios[i]
                        new_box = [current_x, y1, new_x2, y2]
                        new_split_ocr_results.append((new_box, split_text, score))
                        current_x = new_x2
                    continue
                else:
                    new_split_ocr_results.append((box, text, score))
            else:
                new_split_ocr_results.append((box, text, score))
        split_ocr_results = new_split_ocr_results

        new_split_ocr_results = []
        for box, text, score in split_ocr_results:
            new_chars = []
            start = 0
            for i in range(1, len(text)):
                if text[i - 1] in ['。', '.', '！', '？', '?', '!']:  # 判断结尾是否有标点符号
                    if re.search(r"[a-zA-Z]", text[i:i + 1]):
                        if text[i].isupper():
                            new_chars.append(text[start:i])
                            start = i
                    elif re.search(r"[\u4e00-\u9fa5]", text[i:i + 1]):
                        new_chars.append(text[start:i])
                        start = i
            new_chars.append(text[start:])

            if len(new_chars) > 1:
                total_len = len(text)
                len_ratios = [len(s) / total_len for s in new_chars]
                x1, y1, x2, y2 = box
                width = x2 - x1
                current_x = x1
                for i, split_text in enumerate(new_chars):
                    new_x2 = current_x + width * len_ratios[i]
                    new_box = [current_x, y1, new_x2, y2]
                    new_split_ocr_results.append((new_box, split_text, score))
                    current_x = new_x2
            else:
                new_split_ocr_results.append((box, text, score))

        return new_split_ocr_results
