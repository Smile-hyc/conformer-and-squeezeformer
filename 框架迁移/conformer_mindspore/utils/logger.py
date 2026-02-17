import os
import time


class Logger:
    """
    训练日志记录器
    同时输出到控制台和日志文件
    """
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建一个带时间戳的日志文件，防止覆盖
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"train_{timestamp}.log")
        
        print(f"训练日志将保存至: {self.log_path}")
        
    def log(self, message):
        """同时打印到控制台和写入文件"""
        # 添加时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        print(message)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(formatted_message + '\n')
    
    def info(self, message):
        """记录信息级别日志"""
        self.log(f"[INFO] {message}")
    
    def warning(self, message):
        """记录警告级别日志"""
        self.log(f"[WARNING] {message}")
    
    def error(self, message):
        """记录错误级别日志"""
        self.log(f"[ERROR] {message}")
    
    def debug(self, message):
        """记录调试级别日志"""
        self.log(f"[DEBUG] {message}")
    
    def separator(self, char="=", length=50):
        """打印分隔线"""
        self.log(char * length)
    
    def log_config(self, config, title="配置信息"):
        """记录配置字典"""
        self.separator()
        self.log(title)
        self.separator("-")
        self._log_dict(config, indent=0)
        self.separator()
    
    def _log_dict(self, d, indent=0):
        """递归记录字典内容"""
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                self.log(f"{prefix}{key}:")
                self._log_dict(value, indent + 1)
            else:
                self.log(f"{prefix}{key}: {value}")
    
    def log_epoch(self, epoch, total_epochs, train_loss, val_loss, lr, extra_info=None):
        """记录epoch训练信息"""
        msg = (f"Epoch [{epoch}/{total_epochs}] | "
               f"Train Loss: {train_loss:.4f} | "
               f"Val Loss: {val_loss:.4f} | "
               f"LR: {lr:.2e}")
        
        if extra_info:
            for key, value in extra_info.items():
                if isinstance(value, float):
                    msg += f" | {key}: {value:.4f}"
                else:
                    msg += f" | {key}: {value}"
        
        self.log(msg)
    
    def log_step(self, step, total_steps, loss, extra_info=None):
        """记录训练步骤信息"""
        msg = f"Step [{step}/{total_steps}] | Loss: {loss:.4f}"
        
        if extra_info:
            for key, value in extra_info.items():
                if isinstance(value, float):
                    msg += f" | {key}: {value:.4f}"
                else:
                    msg += f" | {key}: {value}"
        
        self.log(msg)