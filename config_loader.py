# config_loader.py - YAML配置加载工具
import yaml
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

# 全局配置对象，加载后全局可用
global_config = None

def load_config():
    """加载yaml配置文件，仅加载一次"""
    global global_config
    if global_config is None:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                global_config = yaml.safe_load(f)
            print(f"> 配置文件加载成功！路径：{CONFIG_FILE}")
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在：{CONFIG_FILE}，请检查文件路径")
        except yaml.YAMLError as e:
            raise SyntaxError(f"配置文件格式错误：{str(e)}")
    return global_config

# 初始化加载配置
load_config()