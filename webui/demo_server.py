#!/usr/bin/env python3
"""LongHorizonUI WebUI Static Server

仿照 Ref_webui 的 demo_server.py 写法：
- 主页：/ -> webui/index.html
- 静态资源：/figure/*, /demo/*

用法：
  python webui/demo_server.py --host 0.0.0.0 --port 5002
"""

from __future__ import annotations

from pathlib import Path
from flask import Flask, send_from_directory


CURRENT_DIR = Path(__file__).parent.absolute()

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(CURRENT_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename: str):
    """直接从 webui/ 目录下提供静态文件访问。"""
    return send_from_directory(CURRENT_DIR, filename)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LongHorizonUI WebUI Static Server")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=5002, help="服务器端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
