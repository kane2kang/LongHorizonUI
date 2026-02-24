"""
logger 用于日志的记录
"""

import platform
import sys
import logging
from datetime import datetime, timezone


log_level_msg_str = {
    logging.DEBUG: "DEBU",
    logging.INFO: "INFO",
    logging.WARN: "WARN",
    logging.ERROR: "ERRO",
    logging.FATAL: "FATL",
}


class Formatter(logging.Formatter):
    """
    日志的格式化
    """

    def __init__(self, color=platform.system().lower() != "windows"):
        # https://stackoverflow.com/questions/2720319/python-figure-out-local-timezone
        self.tz = datetime.now(timezone.utc).astimezone().tzinfo
        self.color = color

    def format(self, record: logging.LogRecord):
        logstr = "[" + datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S") + "] ["
        logstr += log_level_msg_str.get(record.levelno, record.levelname).strip()
        if sys.version_info >= (3, 9):
            fn = record.filename.removesuffix(".py")
        elif record.filename.endswith(".py"):
            fn = record.filename[:-3]
        logstr += f"] {str(record.name).strip()} | {fn} | {str(record.msg)}"
        return logstr


def get_logger(name: str, lv=logging.INFO, remove_exist=True, format_root=False, log_file=None):
    """
    获取一个logger
    :param name: 日志名称
    :param lv: 日志登记
    :param remove_exist: 是否删除已经存在的日志
    :param format_root:
    :param log_file: 日志保存路径
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(lv)
    if remove_exist and logger.hasHandlers():
        logger.handlers.clear()
    if not logger.hasHandlers():
        syslog = logging.StreamHandler()
        syslog.setFormatter(Formatter())
        logger.addHandler(syslog)
    else:
        for h in logger.handlers:
            h.setFormatter(Formatter())

    if log_file is not None:
        print(f"add log to file:{log_file}")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(Formatter())
        logger.addHandler(file_handler)

    if format_root:
        for h in logger.root.handlers:
            h.setFormatter(Formatter())
    return logger
