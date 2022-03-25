import os
import logging
import logging.handlers
import time
from settings import PR


class LogInit:
    __instance = None

    def __init__(self):
        _log_dir = os.path.join(PR, 'logs')
        _log_name = time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.txt'

        self.logger = logging.getLogger(_log_name)
        self.logger.setLevel(logging.DEBUG)
        # 定义handler的输出格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
        if not self.logger.handlers:
            # 创建一个handler，用于写入日志文件, 存 3 个日志，每个 10M 大小
            file_log_handler = logging.handlers.RotatingFileHandler(os.path.join(_log_dir, _log_name), encoding="utf-8",
                                                                    maxBytes=10 * 1024 * 1024, backupCount=3)
            file_log_handler.setLevel(logging.INFO)
            file_log_handler.setFormatter(formatter)
            self.logger.addHandler(file_log_handler)

    @staticmethod
    def set_logger():
        if not LogInit.__instance:
            LogInit.__instance = LogInit()
        return LogInit.__instance


logger = LogInit.set_logger().logger
