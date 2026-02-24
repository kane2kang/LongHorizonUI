import json
import os
import shutil
import uuid
import copy
from datetime import datetime
import cv2
import numpy as np
from ..common import utils, constants, logger


class GraphNode:
    """
    表示有向图中的一个节点，对应于一个UI界面。
    存储截图、描述、语义向量、感知信息等。
    """

    def __init__(self, screenshot_img_path, perception_infos=None, description='', **kwargs):
        """
        初始化一个 GraphNode 对象。

        Args:
            screenshot_img_path (str): 截图图像文件的路径。
            perception_infos (list, optional): 感知信息字典列表。默认为 []。
            description (str, optional): UI 界面的描述。默认为 ''。
            **kwargs: 额外的关键字参数，包括:
                max_text_len (int): 文本序列编码的最大长度。默认为 128。
                max_text_num (int): 要保留的最大文本元素数量。默认为 128。
        """
        self.logger = logger.get_logger(self.__class__.__name__, log_file=kwargs.get("log_file", None))
        self.graph_dir = os.path.join(constants.GRAPH_DIR, constants.GRAPH_NAME)  # 获取图的目录
        os.makedirs(self.graph_dir, exist_ok=True)  # 创建图的目录，如果不存在
        images_dir = os.path.join(self.graph_dir, "images")  # 获取图像目录
        os.makedirs(images_dir, exist_ok=True)  # 创建图像目录，如果不存在

        self.node_name = str(uuid.uuid4())  # 生成节点的唯一 ID
        img_ext = os.path.splitext(screenshot_img_path)[-1]  # 获取图像扩展名
        self.image_path = os.path.join("images", f"{self.node_name}{img_ext}")  # 存储图像的路径，使用节点 ID 作为文件名

        shutil.copy(src=screenshot_img_path, dst=os.path.join(self.graph_dir, self.image_path))  # 复制截图

        self.description = description  # 设置节点描述
        self.perception_infos = copy.deepcopy(perception_infos) if perception_infos else []  # 使用空列表作为默认值
        self.perception_infos_map = {}  # 用于通过 ID 快速查找感知信息
        for pinfo in self.perception_infos:
            if 'id' not in pinfo:
                pid = utils.generate_unique_md5(json.dumps(pinfo))  # 为感知信息生成唯一 ID
                pinfo['id'] = pid
            else:
                pid = pinfo['id']
            self.perception_infos_map[pid] = pinfo  # 构建 ID 映射

        self.max_text_len = kwargs.get("max_text_len", 128)  # 文本编码的最大长度
        self.max_text_num = kwargs.get("max_text_num", 128)  # 要保留的最大文本数量
        self.gen_text_embedding()  # 为节点感知中的文本生成嵌入向量
        self.crop_icons()  # 从截图中裁剪图标

        self.history_actions = []  # 此节点上执行的操作
        self.action_infos = {}  # 通过操作名称存储操作信息
        self.tasks = []  # 与此节点关联的任务

        self.created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 记录创建时间
        self.updated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 记录更新时间

    def update_description(self, description, screen_shot=None):
        """
        更新节点的描述，并可选择更新截图。

        Args:
            description (str): 新的描述。
            screen_shot (str, optional): 新的截图路径。默认为 None。
        """
        self.description = description  # 更新描述
        if screen_shot:
            img_ext = os.path.splitext(screen_shot)[-1]  # 获取图像扩展名
            self.image_path = os.path.join("images", f"{self.node_name}{img_ext}")
            shutil.copy(src=screen_shot, dst=os.path.join(self.graph_dir, self.image_path))  # 复制新的截图
        self.update_time()  # 更新最后修改时间

    def update_perception_infos(self, cur_perception_infos, **kwargs):
        """
        根据新的感知信息更新节点的感知信息。

        Args:
            cur_perception_infos (list): 当前感知信息字典列表。
            **kwargs: 额外的关键字参数，包括:
                box_iou_thred (float): 边界框交并比阈值，用于判断边界框是否相同。默认为 0.6。
                text_iou_thred (float): 文本标记交并比阈值，用于判断文本是否相同。默认为 0.6。
        """
        org_perception_infos = copy.deepcopy(self.perception_infos)  # 复制原始感知信息
        new_perception_infos = []  # 存储新的感知信息

        box_iou_threshold = kwargs.get("box_iou_thred", 0.6)  # 获取边界框 IoU 阈值
        text_iou_threshold = kwargs.get("text_iou_thred", 0.6)  # 获取文本标记 IoU 阈值

        for pinfo in cur_perception_infos:
            cur_box = np.array(pinfo['box'])  # 当前边界框
            org_boxs = []  # 存储原始边界框的列表
            if not org_perception_infos:
                # 如果没有原始感知信息，则添加新的感知信息并继续
                if 'id' not in pinfo:
                    pid = utils.generate_unique_md5(json.dumps(pinfo))  # 生成唯一 ID
                    pinfo['id'] = pid
                new_perception_infos.append(pinfo)
                continue

            for o_pinfo in org_perception_infos:
                org_boxs.append(o_pinfo['box'])  # 存储所有原始边界框
            org_boxs = np.array(org_boxs)
            ious = utils.compute_ious(cur_box, org_boxs)  # 计算 IoU
            max_iou_ind = np.argmax(ious)
            max_iou = ious[max_iou_ind]  # 获取最大 IoU 和对应的索引
            pinfo_ = org_perception_infos[max_iou_ind]  # 获取具有最大 IoU 的感知信息

            if max_iou > box_iou_threshold and pinfo_['type'] == pinfo['type']:
                # 如果重叠足够大且是相同类型的感知信息
                org_perception_infos.pop(max_iou_ind)  # 弹出原始信息，因为我们将合并它
                pinfo_['box'] = pinfo['box']  # 使用新的边界框
                if pinfo_['type'] == 'icon':
                    # 对于图标，保持文本不变
                    if not pinfo_['text'].startswith('AI caption:'):
                        pinfo_['text'] = pinfo['text']
                    new_perception_infos.append(pinfo_)
                else:
                    # 对于文本，检查文本内容是否也相似
                    text_ids = set(utils.encoding.encode(pinfo['text']))
                    text_ids_ = set(utils.encoding.encode(pinfo_['text']))
                    text_iou = len(text_ids_ & text_ids) / (len(text_ids_ | text_ids) + 1e-08)  # 计算文本 IoU
                    if text_iou >= text_iou_threshold:
                        # 如果文本重叠足够大，则合并并使用新文本
                        pinfo_['text'] = pinfo['text']
                        new_perception_infos.append(pinfo_)
                    else:
                        # 如果文本明显不同，则视为新信息
                        if 'id' not in pinfo:
                            pid = utils.generate_unique_md5(json.dumps(pinfo))
                            pinfo['id'] = pid
                        new_perception_infos.append(pinfo)
            else:
                # 如果重叠不够大或它们不是同一类型，则作为新信息添加
                if 'id' not in pinfo:
                    pid = utils.generate_unique_md5(json.dumps(pinfo))
                    pinfo['id'] = pid
                new_perception_infos.append(pinfo)

        self.perception_infos = copy.deepcopy(new_perception_infos)  # 设置新的感知信息
        self.crop_icons()  # 更新感知信息后，重新裁剪图标
        self.perception_infos_map = {}  # 重新创建映射
        for pinfo in self.perception_infos:
            if 'id' not in pinfo:
                pid = utils.generate_unique_md5(json.dumps(pinfo))
                pinfo['id'] = pid
            else:
                pid = pinfo['id']
            self.perception_infos_map[pid] = pinfo  # 根据新的信息构建映射

    def crop_icons(self):
        """
        从截图中裁剪图标图像，并将裁剪后的图像路径存储在感知信息中。
        """
        image = cv2.imread(os.path.join(self.graph_dir, self.image_path))
        if image is None:
            self.logger.error(f"无法读取图像: {os.path.join(self.graph_dir, self.image_path)}")
            return
        h, w = image.shape[:2]  # 获取图像的高和宽
        icons_dir = os.path.join(self.graph_dir, "icons")
        os.makedirs(icons_dir, exist_ok=True)  # 确保图标目录存在
        for pinfo in self.perception_infos:
            if pinfo["type"] == "icon":
                x1, y1, x2, y2 = pinfo["box"]
                ox1 = int(x1 * w)
                ox2 = int(x2 * w)
                oy1 = int(y1 * h)
                oy2 = int(y2 * h)
                icon_image = image[oy1:oy2, ox1:ox2]
                icon_name = f"{self.node_name}-{x1:0.3f}-{y1:0.3f}-{x2:0.3f}-{y2:0.3f}.png"  # 根据边界框坐标生成唯一名称
                cv2.imwrite(os.path.join(icons_dir, icon_name), icon_image)  # 保存裁剪后的图标图像
                pinfo["icon_path"] = os.path.join("icons", icon_name)  # 保存图标图像路径

    def add_icon(self, icon_box, icon_description='', clickable=True, is_clicked=True):
        """
        向节点的感知信息中添加新的图标。

        Args:
            icon_box (tuple): 图标边界框的 (x1, y1, x2, y2) 坐标元组。
            icon_description (str, optional): 图标的描述。默认为 ''。
            clickable (bool, optional): 指示图标是否可点击的标志。默认为 True。
            is_clicked (bool, optional): 指示图标是否被点击的标志。默认为 True。

        Returns:
            dict: 创建的图标感知信息。
        """
        image = cv2.imread(os.path.join(self.graph_dir, self.image_path))
        if image is None:
            self.logger.error(f"无法读取图像: {os.path.join(self.graph_dir, self.image_path)}")
            return None
        h, w = image.shape[:2]  # 获取图像的高和宽
        icons_dir = os.path.join(self.graph_dir, "icons")
        os.makedirs(icons_dir, exist_ok=True)
        pinfo = {
            "type": "icon",
            "text": icon_description,
            "box": (icon_box[0] / w, icon_box[1] / h, icon_box[2] / w, icon_box[3] / h),
            "clickable": clickable,
            "is_clicked": is_clicked
        }
        pid = utils.generate_unique_md5(json.dumps(pinfo))  # 为感知信息生成唯一 ID
        pinfo['id'] = pid
        x1, y1, x2, y2 = pinfo["box"]
        ox1, oy1, ox2, oy2 = icon_box
        icon_image = image[oy1:oy2, ox1:ox2]
        icon_name = f"{self.node_name}-{x1:0.3f}-{y1:0.3f}-{x2:0.3f}-{y2:0.3f}.png"
        cv2.imwrite(os.path.join(icons_dir, icon_name), icon_image)
        pinfo["icon_path"] = os.path.join("icons", icon_name)
        self.perception_infos.append(pinfo)
        self.update_time()
        return pinfo

    def gen_text_embedding(self):
        """
        为节点感知信息中的所有文本元素生成文本嵌入向量。
        存储文本标记 ID 和相对位置。
        """
        self.text_positions = np.ones(shape=(self.max_text_num, 2)) * -10000  # 使用 -10000 表示没有文本
        self.text_ids = np.ones(shape=(self.max_text_num, self.max_text_len)) * -10000  # 使用 -10000 表示没有文本
        self.text_num = 0  # 文本数量
        for i, pinfo in enumerate(self.perception_infos):
            if pinfo["type"] == "text":
                text_ids_ = utils.encoding.encode(pinfo['text'])
                self.text_ids[i, :len(text_ids_)] = text_ids_  # 保存编码后的文本
                x1, y1, x2, y2 = pinfo["box"]
                self.text_positions[i] = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # 保存文本框的中心作为相对位置
                self.text_num += 1

    def add_action(self, action_name, action_info):
        """
        向节点添加一个操作。

        Args:
            action_name (str): 操作的名称。
            action_info (dict): 关于此操作的额外信息
        """
        self.action_infos[action_name] = action_info
        self.history_actions.append(action_name)  # 记录执行的操作
        self.update_time()

    def update_time(self):
        """
        使用当前时间更新 updated_time 属性。
        """
        self.updated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
