# 新建一个文件 recorder.py
import sys
import subprocess
import time
import os
import signal


def main():
    if len(sys.argv) < 3:
        print("Usage: recorder.py <device_id> <output_path>")
        sys.exit(1)

    device_id = sys.argv[1]
    output_path = sys.argv[2]

    cmd = [
        "scrcpy",
        "-s", device_id,
        "--record", output_path,
        "--no-window",
    ]

    # 启动录制
    process = subprocess.Popen(cmd)

    # 等待外部终止信号
    try:
        process.wait()
    except KeyboardInterrupt:
        # 当收到CTRL+C时正常结束
        pass

    print(f"视频已保存到: {output_path}")


if __name__ == "__main__":
    main()
