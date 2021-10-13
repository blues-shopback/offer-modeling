"""running config."""
from models.config_utils import BaseConfig


logger = None


class RunningConfig(BaseConfig):
    @staticmethod
    def get_keys():
        return ["batch_size", "clip", "max_seq_len", "weight_decay",
                "learning_rate"]
