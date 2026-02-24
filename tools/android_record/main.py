import pdb
import platform
import shutil
import sys
import threading
import scrcpy
from PySide6.QtGui import QMouseEvent, QImage, QPixmap, QKeyEvent, QFont
from adbutils import adb
from PySide6.QtCore import *
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, \
    QCheckBox, QLabel, QGridLayout, QSpacerItem, QSizePolicy, QLineEdit, QInputDialog
from PySide6.QtCore import Qt
from numpy import ndarray
import platform

import time
import os
import json
import subprocess

# 创建QApplication对象
if not QApplication.instance():
    app = QApplication([])
else:
    app = QApplication.instance()

items = [i.serial for i in adb.device_list()]  # 设备列表
client_dict = {}  # 设备scrcpy客服端字典
# 为所有设备建立scrcpy服务
for i in items:
    client_dict[i] = scrcpy.Client(device=i, bitrate=10000000)
    time.sleep(1.0)


def thread_ui(func, *args):
    """
    开启一个新线程任务\n
    :param func: 要执行的线程函数;
    :param args: 函数中需要传入的参数 Any
    :return:
    """
    t = threading.Thread(target=func, args=args)  # 定义新线程
    t.setDaemon(True)  # 开启线程守护
    t.start()  # 执行线程


class SignThread(QThread):
    """信号线程"""

    def __new__(cls, parent: QWidget, func, *types: type):
        cls.__update_date = Signal(*types, name=func.__name__)  # 定义信号(*types)一个信号中可以有一个或多个类型的数据(int,str,list,...)
        return super().__new__(cls)  # 使用父类__new__方法创建SignThread实例对象

    def __init__(self, parent: QWidget, func, *types: type):
        """
        信号线程初始化\n
        :param parent: 界面UI控件
        :param func: 信号要绑定的方法
        :param types: 信号类型,可以是一个或多个(type,...)
        """
        super().__init__(parent)  # 初始化父类
        self.__update_date.connect(func)  # 绑定信号与方法

    def send_sign(self, *args):
        """
        使用SignThread发送信号\n
        :param args: 信号的内容
        :return:
        """
        self.__update_date.emit(*args)  # 发送信号元组(type,...)


class RecordWindow(QWidget):
    """UI界面"""

    def __init__(self, save_dir="./data"):
        """UI界面初始化"""
        super().__init__()  # 初始化父级
        self.setWindowTitle('QAgent Android 屏幕录制')  # 设置窗口标题
        self.max_width = 600  # 设置手机投屏宽度
        # 定义元素
        self.check_box = QCheckBox("控制所有设备")  # 定义是否控制所有设备选择框
        self.back_button = QPushButton("返回")  # 定义返回键
        self.back_button.setFixedWidth(60)
        self.home_button = QPushButton("主页")  # 定义home键
        self.home_button.setFixedWidth(60)
        self.recent_button = QPushButton("后台")  # 定义最近任务键
        self.recent_button.setFixedWidth(60)
        self.refresh_button = QPushButton("新任务")
        self.refresh_button.setFixedWidth(70)
        self.input_text_button = QPushButton("输入文本")  # 新增输入文本按钮
        self.input_text_button.setFixedWidth(70)
        self.recording_process = None  # 存储录制进程
        self.temp_video_path = None  # 临时视频路径
        # self.input_field = QLineEdit(self)#输入框
        # self.input_field.setFixedWidth(200)
        # self.check_label = QLabel(self)#判断框
        # self.check_label.setFixedWidth(20)  # 设置固定宽度
        # self.confirm_button = QPushButton("确认", self)#确认按钮
        # self.confirm_button.setFixedWidth(50)
        self.video = QLabel("设备屏幕信息加载......")  # 定义手机投屏控制标签
        self.video.setStyleSheet("border-width: 3px;border-style: solid;border-color: black;")  # 定义投屏标签样式
        self.task_id = str(time.time())
        self.task_label = QLabel(f"任务ID:{self.task_id}", self)
        self.task_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.font = QFont()
        self.font.setPointSize(12)
        self.task_label.setFont(self.font)
        self.video_list = []  # 定义手机投屏标签列表
        for i in items:
            self.video_list.append(QLabel(i))  # 把投屏标签加入列表

        self.mouse_thread = SignThread(self, self.mouse_exe, int, int, int)
        self.main_thread = SignThread(self, self.main_exe, ndarray)
        self.on_thread = SignThread(self, self.on_exe, int, int, ndarray)
        time.sleep(1.0)

        if len(items) > 0:
            self.now_device = items[0]
            self.now_client = client_dict[items[0]]
            self.now_client.add_listener(scrcpy.EVENT_FRAME, self.main_frame)
            time.sleep(1)

        self.main_layout = QHBoxLayout(self)  # 定义主布局容器
        self.frame_layout = QVBoxLayout()  # 定义投屏操控框容器
        self.button_layout = QHBoxLayout()
        self.button_layout2 = QHBoxLayout()  # 新增第二行按钮布局
        # self.button_layout1 = QHBoxLayout()
        self.device_layout = QVBoxLayout()  # 定义投屏容器
        self.list_layout = QGridLayout()  # 定义投屏列表布局容器
        self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)  # 弹性空间
        self.device_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)  # 弹性空间
        self.v_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)  # 弹性空间
        # 页面布局
        self.main_layout.addLayout(self.frame_layout)
        self.main_layout.addLayout(self.device_layout)
        self.main_layout.addItem(self.v_spacer)
        self.frame_layout.addWidget(self.task_label)
        self.frame_layout.addWidget(self.video)
        self.frame_layout.addLayout(self.button_layout)
        self.frame_layout.addLayout(self.button_layout2)  # 添加第二行按钮布局
        # self.frame_layout.addItem(self.button_layout1)
        # self.frame_layout.addWidget(self.check_box)
        self.frame_layout.addItem(self.spacer)
        self.button_layout.addWidget(self.back_button)
        self.button_layout.addWidget(self.home_button)
        self.button_layout.addWidget(self.recent_button)
        self.button_layout2.addWidget(self.input_text_button)
        self.button_layout2.addWidget(self.refresh_button)  # 添加输入文本按钮到第二行
        # self.button_layout1.addWidget(self.input_field)
        # self.button_layout1.addWidget(self.check_label)
        # self.button_layout1.addWidget(self.confirm_button)
        # self.device_layout.addLayout(self.list_layout)
        self.device_layout.addItem(self.device_spacer)
        # 交互事件
        # self.confirm_button.clicked.connect(self.on_confirm)
        self.back_button.clicked.connect(self.click_key(scrcpy.KEYCODE_BACK))
        self.home_button.clicked.connect(self.click_key(scrcpy.KEYCODE_HOME))
        self.recent_button.clicked.connect(self.click_key(scrcpy.KEYCODE_APP_SWITCH))
        self.refresh_button.clicked.connect(self.click_key(scrcpy.KEYCODE_UNKNOWN))
        self.input_text_button.clicked.connect(self.input_text)  # 连接输入文本按钮事件
        self.video.mousePressEvent = self.mouse_event(scrcpy.ACTION_DOWN)
        self.video.mouseMoveEvent = self.mouse_event(scrcpy.ACTION_MOVE)
        self.video.mouseReleaseEvent = self.mouse_event(scrcpy.ACTION_UP)
        self.keyPressEvent = self.on_key_event(scrcpy.ACTION_DOWN)
        self.keyReleaseEvent = self.on_key_event(scrcpy.ACTION_UP)

        self.last_action = None
        self.iter = 1
        self.xy1 = (float("inf"), float("inf"))
        self.xy2 = (float("inf"), float("inf"))
        self.action_infos = []
        self.image = None
        self.last_time = None
        self.save_dir = save_dir

        # 自动开始录制视频
        # self.start_recording(True)
        # time.sleep(1)
        # self.stop_recording(True)
        # shutil.rmtree(os.path.join(self.save_dir, self.task_id))
        # task_id_tem = time.time()
        # self.task_id = str(task_id_tem)
        # self.task_label.setText(f"任务ID:{task_id_tem}")
        # self.start_recording()

    def input_text(self):
        """
        输入文本功能
        """
        text, ok = QInputDialog.getText(self, "输入文本", "请输入要发送的文本:")
        if ok and text:
            # 使用adb输入文本
            if self.check_box.isChecked():
                for i in client_dict:
                    adb.device(i).shell(f'input text "{text}"')
            else:
                adb.device(self.now_device).shell(f'input text "{text}"')

            # 保存截图和记录操作
            task_save_dir = os.path.join(self.save_dir, self.task_id)
            os.makedirs(task_save_dir, exist_ok=True)
            task_image_dir = os.path.join(task_save_dir, "screenshot")
            os.makedirs(task_image_dir, exist_ok=True)
            screenshot_img_path = os.path.join(task_image_dir, f"{self.iter:03d}.png")
            if self.image is not None:
                print(screenshot_img_path)
                self.image.save(screenshot_img_path, "PNG", 100)

            # 记录操作
            self.action_infos.append({
                "timestamp": time.time(),
                "action": f"Input Text: {text}",
                "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)
            })
            self.iter += 1

    def start_recording(self, first_run=False):
        """
        开始录制视频
        """
        while not self.now_device or not adb.device_list():
            print("No device available, retrying in 1 second...")
            time.sleep(1)

        # 创建本地保存目录
        task_save_dir = os.path.join(self.save_dir, self.task_id)
        os.makedirs(task_save_dir, exist_ok=True)
        video_path = os.path.join(task_save_dir, "record.mp4")
        if platform.system().lower() == "windows":
            if first_run:
                cmd = [
                    "scrcpy",
                    "--record", video_path,  # 直接录制到本地文件
                    # "--capture-orientation", "0",  # 试试 0, 1, 2, 3
                    "--no-window",
                    "-s", self.now_device  # 指定设备
                ]
                self.recording_process = subprocess.Popen(cmd)
            else:
                self.recording_process = subprocess.Popen([
                    "python", "video_recorder.py",
                    self.now_device, video_path
                ])
        else:
            cmd = [
                "scrcpy",
                "--record", video_path,  # 直接录制到本地文件
                # "--capture-orientation", "0",  # 试试 0, 1, 2, 3
                "--no-window",
                "-s", self.now_device  # 指定设备
            ]
            self.recording_process = subprocess.Popen(cmd)

        print(f"开始录制视频到: {video_path}")

    def stop_recording(self, first_run=False):
        """
        停止录制视频
        """
        if self.recording_process:
            import signal
            if platform.system().lower() == "windows":  # Windows
                if first_run:
                    self.recording_process.terminate()
                else:
                    # 向子进程发送CTRL+C
                    self.recording_process.send_signal(signal.CTRL_C_EVENT)
            else:
                self.recording_process.terminate()

            # 等待进程结束
            self.recording_process = None
            time.sleep(1)

            print(f"视频录制已停止并保存到: {self.temp_video_path}")

    def click_key(self, key_value: int):
        """
        按键事件\n
        :param key_value: 键值
        :return:
        """

        def key_event():
            if key_value == scrcpy.KEYCODE_UNKNOWN:
                task_save_dir = os.path.join(self.save_dir, self.task_id)
                os.makedirs(task_save_dir, exist_ok=True)
                task_image_dir = os.path.join(task_save_dir, "screenshot")
                os.makedirs(task_image_dir, exist_ok=True)
                screenshot_img_path = os.path.join(task_image_dir, f"{self.iter:03d}.png")
                if self.image is not None:
                    self.image.save(screenshot_img_path, "PNG", 100)

                task_save_dir = os.path.join(self.save_dir, self.task_id)
                os.makedirs(task_save_dir, exist_ok=True)
                with open(os.path.join(task_save_dir, "actions.json"), "w") as fw:
                    json.dump(self.action_infos, fw)
                while True:
                    try:
                        # 停止当前录制并保存视频
                        self.stop_recording()
                        self.last_action = None
                        self.iter = 1
                        self.xy1 = (float("inf"), float("inf"))
                        self.xy2 = (float("inf"), float("inf"))
                        self.action_infos = []
                        self.image = None
                        self.last_time = None
                        task_id_tem = time.time()
                        self.task_id = str(task_id_tem)
                        self.task_label.setText(f"任务ID:{task_id_tem}")

                        # 开始新的录制
                        self.start_recording()
                    except KeyboardInterrupt as e:
                        print("KeyboardInterrupt happened! Start recording again!")
                        time.sleep(1)
                        continue
                    break
            elif key_value == scrcpy.KEYCODE_HOME:
                self.now_client.control.keycode(key_value, scrcpy.ACTION_DOWN)
                self.now_client.control.keycode(key_value, scrcpy.ACTION_UP)

                task_save_dir = os.path.join(self.save_dir, self.task_id)
                os.makedirs(task_save_dir, exist_ok=True)
                task_image_dir = os.path.join(task_save_dir, "screenshot")
                os.makedirs(task_image_dir, exist_ok=True)
                screenshot_img_path = os.path.join(task_image_dir, f"{self.iter:03d}.png")
                if self.image is not None:
                    print(screenshot_img_path)
                    self.image.save(screenshot_img_path, "PNG", 100)
                self.action_infos.append({
                    "timestamp": time.time(),
                    "action": f"Home",
                    "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)
                })
                self.iter += 1
            elif key_value == scrcpy.KEYCODE_BACK:
                self.now_client.control.keycode(key_value, scrcpy.ACTION_DOWN)
                self.now_client.control.keycode(key_value, scrcpy.ACTION_UP)

                task_save_dir = os.path.join(self.save_dir, self.task_id)
                os.makedirs(task_save_dir, exist_ok=True)
                task_image_dir = os.path.join(task_save_dir, "screenshot")
                os.makedirs(task_image_dir, exist_ok=True)
                screenshot_img_path = os.path.join(task_image_dir, f"{self.iter:03d}.png")
                if self.image is not None:
                    print(screenshot_img_path)
                    self.image.save(screenshot_img_path, "PNG", 100)
                self.action_infos.append({
                    "timestamp": time.time(),
                    "action": f"Go Back",
                    "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)
                })
                self.iter += 1
            elif key_value == scrcpy.KEYCODE_APP_SWITCH:
                self.now_client.control.keycode(key_value, scrcpy.ACTION_DOWN)
                self.now_client.control.keycode(key_value, scrcpy.ACTION_UP)

                task_save_dir = os.path.join(self.save_dir, self.task_id)
                os.makedirs(task_save_dir, exist_ok=True)
                task_image_dir = os.path.join(task_save_dir, "screenshot")
                os.makedirs(task_image_dir, exist_ok=True)
                screenshot_img_path = os.path.join(task_image_dir, f"{self.iter:03d}.png")
                if self.image is not None:
                    print(screenshot_img_path)
                    self.image.save(screenshot_img_path, "PNG", 100)
                self.action_infos.append({
                    "timestamp": time.time(),
                    "action": f"View recent apps",
                    "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)
                })
                self.iter += 1

        return key_event

    def main_frame(self, frame: ndarray):
        """
        监听设备屏幕数据,设置控制窗口图像\n
        :param frame: 图像帧
        :return:
        """
        if frame is not None:
            self.main_thread.send_sign(frame)

    def main_exe(self, frame):
        """
        主控屏幕图像设置\n
        :param frame: 图像帧
        :return:
        """
        ratio = self.max_width / max(self.now_client.resolution)
        image = QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.shape[1] * 3,
            QImage.Format_BGR888,
        )
        pix = QPixmap(image)
        pix.setDevicePixelRatio(1 / ratio)
        self.video.setPixmap(pix)
        self.image = image

    def on_frame(self, num: int, client: scrcpy.Client):
        """
        监听设备屏幕数据,设置小窗口图像\n
        :param num: 设备投屏序号
        :param client: scrcpy服务
        :return:
        """

        def client_frame(frame: ndarray):
            if frame is not None:
                self.on_thread.send_sign(num, max(client.resolution), frame)

        return client_frame

    def on_exe(self, num: int, resolution: int, frame):
        """
        小窗口图像设置\n
        :param num: 设备投屏序号
        :param resolution: 设备宽度
        :param frame: 图像帧
        :return:
        """
        ratio = 300 / resolution
        image = QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.shape[1] * 3,
            QImage.Format_BGR888,
        )
        pix = QPixmap(image)
        pix.setDevicePixelRatio(1 / ratio)
        self.video_list[num].setPixmap(pix)

    def mouse_event(self, action=scrcpy.ACTION_DOWN):
        """
        鼠标事件\n
        :param action: 事件类型
        :return: 对应的事件函数
        """

        def event(evt: QMouseEvent):
            focused_widget = QApplication.focusWidget()
            if focused_widget is not None:
                focused_widget.clearFocus()
            ratio = self.max_width / max(self.now_client.resolution)
            self.mouse_thread.send_sign(evt.position().x() / ratio, evt.position().y() / ratio, action)

        return event

    def mouse_exe(self, x: int, y: int, action: int):
        """
        执行鼠标事件\n
        :param x: x坐标
        :param y: y坐标
        :param action: 事件类型
            0:按下
            1：松开
            2：移动
        :return:
        """
        if self.check_box.isChecked():
            for i in client_dict:
                client_dict[i].control.touch(x, y, action)
        else:
            self.now_client.control.touch(x, y, action)

        task_save_dir = os.path.join(self.save_dir, self.task_id)
        os.makedirs(task_save_dir, exist_ok=True)
        task_image_dir = os.path.join(task_save_dir, "screenshot")
        os.makedirs(task_image_dir, exist_ok=True)
        action_info_ = None
        if action == 0:
            screenshot_img_path = os.path.join(task_image_dir, f"{self.iter:03d}.png")
            self.last_time = time.time()
            self.xy1 = (x, y)
            if self.image is not None:
                print(screenshot_img_path)
                self.image.save(screenshot_img_path, "PNG", 100)
            self.iter += 1
        elif action == 1:
            screenshot_img_path = os.path.join(task_image_dir, f"{self.iter - 1:03d}.png")
            self.xy2 = (x, y)
            if self.last_action == 0:
                tem_time = time.time()
                if tem_time - self.last_time < 1.5:
                    action_info_ = {
                        "timestamp": time.time(),
                        "action": f"Tap {self.xy2}",
                        "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)

                    }
                else:
                    action_info_ = {
                        "timestamp": time.time(),
                        "action": f"Press {self.xy2}",
                        "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)
                    }
            elif self.last_action == 2:
                self.xy2 = (x, y)
                x1, y1 = self.xy1
                x2, y2 = self.xy2
                tem_time = time.time()
                if max(abs(x1 - x2), abs(y1 - y2)) < 20:
                    # 如果距离很短，则认为是一个fake swipe的操作
                    if tem_time - self.last_time < 1.5:
                        action_info_ = {
                            "timestamp": time.time(),
                            "action": f"Tap {self.xy1}",
                            "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)

                        }
                    else:
                        action_info_ = {
                            "timestamp": time.time(),
                            "action": f"Press {self.xy1}",
                            "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)
                        }
                else:
                    if tem_time - self.last_time < 1.5:
                        action_info_ = {
                            "timestamp": time.time(),
                            "action": f"Swipe {self.xy1}, {self.xy2}",
                            "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)
                        }
                    else:
                        action_info_ = {
                            "timestamp": time.time(),
                            "action": f"Drag {self.xy1}, {self.xy2}",
                            "screenshot": os.path.relpath(screenshot_img_path, task_save_dir)
                        }
            if action_info_:
                self.action_infos.append(action_info_)

        self.last_action = action

    def on_key_event(self, action=scrcpy.ACTION_DOWN):
        """
        键盘按键事件\n
        :param action: 事件类型
        :return: 对应的事件函数
        """

        def handler(evt: QKeyEvent):
            code = self.key_code(evt.key())
            if code != -1:
                if self.check_box.isChecked():
                    for i in client_dict:
                        client_dict[i].control.keycode(code, action)
                else:
                    self.now_client.control.keycode(code, action)

        return handler

    @staticmethod
    def key_code(code):
        """
        Map qt keycode ti android keycode
        Args:
            code: qt keycode
            android keycode, -1 if not founded
        """
        if 48 <= code <= 57:
            return code - 48 + 7
        if 65 <= code <= 90:
            return code - 65 + 29
        if 97 <= code <= 122:
            return code - 97 + 29

        hard_code = {
            35: scrcpy.KEYCODE_POUND,
            42: scrcpy.KEYCODE_STAR,
            44: scrcpy.KEYCODE_COMMA,
            46: scrcpy.KEYCODE_PERIOD,
            32: scrcpy.KEYCODE_SPACE,
            16777219: scrcpy.KEYCODE_DEL,
            16777248: scrcpy.KEYCODE_SHIFT_LEFT,
            16777220: scrcpy.KEYCODE_ENTER,
            16777217: scrcpy.KEYCODE_TAB,
            16777249: scrcpy.KEYCODE_CTRL_LEFT,
            16777235: scrcpy.KEYCODE_DPAD_UP,
            16777237: scrcpy.KEYCODE_DPAD_DOWN,
            16777234: scrcpy.KEYCODE_DPAD_LEFT,
            16777236: scrcpy.KEYCODE_DPAD_RIGHT,
        }
        if code in hard_code:
            return hard_code[code]

        print(f"Unknown keycode: {code}")
        return -1

    def closeEvent(self, _):
        """窗口关闭事件"""
        # 停止录制并保存视频
        if self.recording_process:
            self.stop_recording()

        for i in client_dict:
            client_dict[i].stop()  # 关闭scrcpy服务


def main():
    for i, key in enumerate(client_dict):
        thread_ui(client_dict[key].start)  # 给每一台设备单独开启一个scrcpy服务线程
    widget = RecordWindow()  # 实例化UI线程
    widget.resize(800, 800)  # 设置窗口大小
    widget.show()  # 展示窗口
    sys.exit(app.exec())  # 持续刷新窗口


if __name__ == '__main__':
    main()
