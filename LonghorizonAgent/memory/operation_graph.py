import os
import json
import uuid
import pickle
from datetime import datetime
from typing import Dict, List

from LonghorizonAgent.agent.graph_node import GraphNode        # ← 即您上文给出的 GraphNode
from LonghorizonAgent.common import logger

_log = logger.get_logger("OperationGraph")


class ActionEdge:
    """
    图中的“边”，表示一次具体操作。保存：
    - src / dst  两端节点 id
    - exec_info  LLM 生成或原始 action 信息
    """
    def __init__(self, src_node: GraphNode, dst_node: GraphNode, exec_info: dict):
        self.edge_id = str(uuid.uuid4())
        self.src = src_node.node_name
        self.dst = dst_node.node_name
        self.exec_info = exec_info
        self.created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        d = self.__dict__.copy()
        # 让 json 序列化时不要深拷贝 GraphNode——只保留 id
        d.pop("src_node", None)
        d.pop("dst_node", None)
        return d


class OperationGraph:
    """
    包装一个有向图：nodes(id→GraphNode) + edges(List[ActionEdge])
    提供 add / save / load 等基础操作
    """
    def __init__(self, graph_dir: str, graph_name: str = "operation_graph"):
        self.graph_dir = os.path.join(graph_dir, graph_name)
        os.makedirs(self.graph_dir, exist_ok=True)

        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[ActionEdge] = []

    # ---------- 节点 ----------
    def add_node(self, node: GraphNode):
        if node.node_name not in self.nodes:
            self.nodes[node.node_name] = node

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id, None)

    # ---------- 边 ----------
    def add_edge(self, edge: ActionEdge):
        self.edges.append(edge)

    def save(self):
        """
        1. 调用每个 GraphNode 已有的落盘逻辑（它们会把截图 / icon 存进各自目录）
        2. 将整个图对象序列化并保存为 pickle 格式
        """
        _log.info("Saving operation graph ...")

        # 使用 pickle 序列化整个 OperationGraph 对象
        graph_path = os.path.join(self.graph_dir, "operation_graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(self, f)  # 序列化整个图对象

        _log.info(f"Graph saved to {graph_path}")

    @classmethod
    def load(cls, graph_path: str) -> "OperationGraph":
        """
        从 pickle 文件中加载操作图（反序列化）
        """
        _log.info(f"Loading operation graph from {graph_path}...")

        # 使用 pickle 加载整个图对象
        with open(graph_path, "rb") as f:
            og = pickle.load(f)  # 反序列化整个图对象

        _log.info(f"Loaded graph from {graph_path}")
        return og

    # 存储为json格式
    # def save(self):
    #     """
    #     1. 调用每个 GraphNode 已有的落盘逻辑（它们会把截图 / icon 存进各自目录）
    #     2. 把节点 meta + edges meta 写到一个 json
    #     """
    #     _log.info("Saving operation graph ...")
    #     nodes_meta = {
    #         nid: {
    #             "image_path": n.image_path,          # 相对路径，GraphNode 已存好
    #             "description": n.description,
    #             "perception_infos": n.perception_infos,
    #             "created_time": n.created_time,
    #             "updated_time": n.updated_time
    #         }
    #         for nid, n in self.nodes.items()
    #     }
    #     edges_meta = [e.to_dict() for e in self.edges]
    #
    #     with open(os.path.join(self.graph_dir, "operation_graph.json"),
    #               "w", encoding="utf-8") as f:
    #         json.dump({"nodes": nodes_meta, "edges": edges_meta},
    #                   f, indent=2, ensure_ascii=False)
    #
    # @classmethod
    # def load(cls, graph_path: str) -> "OperationGraph":
    #     """
    #     需要“回放”轨迹
    #     图片 / icon 保存在各自目录
    #     """
    #     with open(graph_path, "r", encoding="utf-8") as f:
    #         raw = json.load(f)
    #
    #     graph_dir = os.path.dirname(graph_path)
    #     og = cls(graph_dir=graph_dir)
    #     # 1) 还原节点
    #     for nid, meta in raw["nodes"].items():
    #         node = GraphNode(
    #             screenshot_img_path=os.path.join(graph_dir, meta["image_path"]),
    #             perception_infos=meta["perception_infos"],
    #             description=meta["description"]
    #         )
    #         node.node_name = nid         # 覆盖掉随机 uuid，保持一致
    #         node.created_time = meta["created_time"]
    #         node.updated_time = meta["updated_time"]
    #         og.nodes[nid] = node
    #     # 2) 还原边
    #     og.edges = [
    #         ActionEdge(
    #             src_node=og.nodes[e["src"]],
    #             dst_node=og.nodes[e["dst"]],
    #             exec_info=e["exec_info"]
    #         )
    #         for e in raw["edges"]
    #     ]
    #     return og