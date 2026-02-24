# LonghorizonAgent/common/constants.py   （精简版）
import os
from pathlib import Path

# 工程根目录
PROJECT_DIR: str = Path(__file__).resolve().parent.parent.as_posix()

# ──────────── 日志目录 ────────────
# logger.get_logger(...) 里会用到
LOG_DIR: str = os.getenv("QAGENT_LOG_DIR", '')
if not LOG_DIR:
    LOG_DIR = os.path.join(PROJECT_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ──────────── UI / Operation 图资源根目录 ────────────
# GraphNode 会在此目录下再建  <GRAPH_NAME>/<images|icons|…>
GRAPH_DIR: str = os.getenv("QAGENT_GRAPH_DIR", '')
if not GRAPH_DIR:
    GRAPH_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "data", "graphs"))
os.makedirs(GRAPH_DIR, exist_ok=True)

# 默认图名字；可以在运行时被环境变量或代码覆盖
GRAPH_NAME: str = os.getenv("QAGENT_GRAPH_NAME", "android_general")