import os
import time

class Logger:
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建一个带时间戳的日志文件，防止覆盖
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"train_{timestamp}.log")
        
        print(f"训练日志将保存至: {self.log_path}")
        
    def log(self, message):
        """同时打印到控制台和写入文件"""
        print(message)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')