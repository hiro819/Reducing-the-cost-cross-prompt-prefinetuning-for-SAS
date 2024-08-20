from dataclasses import dataclass, field
import dataclasses
import pathlib
import yaml
import inspect
import numpy as np
import torch


class YAML:
    def save(self, config_path: pathlib.Path):
        """ Export config as YAML file """
        assert config_path.parent.exists(), f'directory {config_path.parent} does not exist'

        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, pathlib.Path):
                    data[key] = str(val)
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path, 'w') as f:
            yaml.dump(convert_dict(dataclasses.asdict(self)), f)

    @classmethod
    def load(cls, config_path: pathlib.Path):
        """ Load config from YAML file """
        assert config_path.exists(), f'YAML config {config_path} does not exist'

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if child_class == pathlib.Path:
                    data[key] = pathlib.Path(val)
                if inspect.isclass(child_class) and issubclass(child_class, YAML):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # recursively convert config item to YamlConfig
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)

    def update(self, new: dict):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class Config(YAML):
    train_path: str = ''
    dev_path: str = ''
    test_path: str = ''
    model_dir: str = "data/model"
    eval_dir: str = "data/eval"
    data_num: int = 50
    prompt: str = ""
    item: str = ''
    input_max_len: int = 512
    target_max_len: int = 512

    use_criteria: bool = False
    adaptive_pretrain: bool = False

    model_name_or_path:str ='cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer_name_or_path:str ='cl-tohoku/bert-base-japanese-whole-word-masking'

    learning_rate:float = 3e-4
    weight_decay: float =0.0
    adam_epsilon: float =1e-8
    warmup_steps:float =0
    gradient_accumulation_steps:int =1
    max_score: int = 3

    train_batch_size:int = 8
    eval_batch_size:int = 8
    num_train_epochs:int = 4

    n_gpu:int =0
    early_stop_callback: bool =False
    fp_16: bool =False
    opt_level:str ='O0'
    max_grad_norm:float =1.0
    seed:int = 42
    use_wandb: str = False
    wandb_project: str = ''
    wandb_name: str = ''