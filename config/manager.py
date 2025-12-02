import yaml
import os

class ConfigManager:
    _instance = None

    def __new__(cls, config_path="config/config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split('.')
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
