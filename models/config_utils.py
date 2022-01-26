import os
import json
from pprint import pformat


def create_json_template(_class, path=None):
    """Create template json file."""
    json_data = {}
    for key in _class.get_keys():
        json_data[key] = 0

    if path:
        with open(path, "w") as f:
            json.dump(json_data, f, indent=4, sort_keys=True)
    else:
        with open(_class.__name__ + ".json", "w") as f:
            json.dump(json_data, f, indent=4, sort_keys=True)


class BaseConfig:
    """Base model config."""

    def __init__(self, json_path=None, data_dict=None):
        assert json_path is not None or data_dict is not None

        self.keys = self.get_keys()

        if json_path is not None:
            self.init_from_json(json_path)
        elif data_dict is not None:
            self.init_from_data(data_dict)

    @classmethod
    def generate_template(cls, path):
        keys = cls.get_keys()
        data = {}
        for key in keys:
            data[key] = None

        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(path, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)

        return data

    @staticmethod
    def get_keys():
        raise NotImplementedError

    def init_from_data(self, data_dict):
        for key in self.keys:
            setattr(self, key, data_dict[key])

    def init_from_json(self, json_path):
        with open(json_path) as f:
            json_data = json.load(f)
            for key in self.keys:
                setattr(self, key, json_data[key])

    def to_json(self, json_path):
        """Save to a json file."""
        json_data = {}
        for key in self.keys:
            json_data[key] = getattr(self, key)

        json_dir = os.path.dirname(json_path)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4, sort_keys=True)

    def format_params(self):
        """Return params as string."""
        json_data = {}
        for key in self.keys:
            json_data[key] = getattr(self, key)

        return pformat(json_data)


class OfferDatasetConfig(BaseConfig):
    @staticmethod
    def get_keys():
        return ["amazon_cate_path", "amazon_wo_cate_path", "amazon_cate_size",
                "catch_cate_path", "catch_cate_size",
                "mydeal_cate_path", "mydeal_wo_cate_path", "mydeal_cate_size"]


class BertConfig(BaseConfig):
    @staticmethod
    def get_keys():
        return ["n_layer", "d_model", "n_head", "d_head", "d_inner", "d_embed",
                "dropout", "dropatt", "pre_ln", "inp_cate_num", "n_token"]


class PriceNetConfig(BaseConfig):
    @staticmethod
    def get_keys():
        return ["category_num", "pre_act_fn", "post_act_fn"]


class FinConfig(BaseConfig):
    @staticmethod
    def get_keys():
        return ["n_layer", "d_model", "n_head", "d_head", "d_inner", "n_ticker", "d_embed",
                "dropout", "dropatt", "pre_ln"]


class XLNetConfig(BaseConfig):
    """XLNetConfig contains hyperparameters that are specific to a model checkpoint;
    i.e., these hyperparameters should be the same between
    pretraining and finetuning.

    The following hyperparameters are defined:
        n_layer: int, the number of layers.
        d_model: int, the hidden size.
        n_head: int, the number of attention heads.
        d_head: int, the dimension size of each attention head.
        d_inner: int, the hidden size in feed-forward layers.
        ff_activation: str, "relu" or "gelu".
        untie_r: bool, whether to untie the biases in attention.
        n_token: int, the vocab size.
        pre_ln: bool, whether use pre-layer_norm.
        pos_len: int, the max length of position embedding.
    """
    @staticmethod
    def get_keys():
        return ["n_layer", "d_model", "n_head", "d_head", "d_inner",
                "ff_activation", "untie_r", "n_token", "pre_ln"]


class XLNetSmConfig(BaseConfig):
    """XLNetConfig contains hyperparameters that are specific to a model checkpoint;
    i.e., these hyperparameters should be the same between
    pretraining and finetuning.

    The following hyperparameters are defined:
        n_layer: int, the number of layers.
        d_model: int, the hidden size.
        n_head: int, the number of attention heads.
        d_head: int, the dimension size of each attention head.
        d_inner: int, the hidden size in feed-forward layers.
        ff_activation: str, "relu" or "gelu".
        untie_r: bool, whether to untie the biases in attention.
        n_token: int, the vocab size.
        pre_ln: bool, whether use pre-layer_norm.
        pos_len: int, the max length of position embedding.
    """
    @staticmethod
    def get_keys():
        return ["n_layer", "d_model", "n_head", "d_head", "d_inner",
                "ff_activation", "untie_r", "n_token", "pre_ln", "d_embed"]
