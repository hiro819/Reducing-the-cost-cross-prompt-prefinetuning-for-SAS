import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from os import path
import sys

sys.path.append("..")
import json

class Util:
    def __init__(self):
        pass

    @staticmethod
    def convert_to_original_score(scores: torch.tensor, max_score: int):
        org_scal_scores = scores * max_score
        org_scal_scores = torch.clip(org_scal_scores, 0, max_score)
        int_scores = torch.round(org_scal_scores)

        if int_scores.ndim == 0:
            int_scores = [int_scores.item()]
        else:
            int_scores = int_scores.tolist()

        return int_scores

    @staticmethod
    def get_callbacks(model_dir, adaptive_pretrain=False):
        if adaptive_pretrain:
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=path.join(model_dir, 'checkpoint'),
                filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
                save_top_k=-1,
                mode='min')
        else:
            checkpoint_callback = ModelCheckpoint(
            monitor = 'val_qwk',
            dirpath = path.join(model_dir,'checkpoint'),
            filename = 'checkpoint-{epoch:02d}-{val_qwk:.2f}',
            save_top_k=-1,
            mode='max')

        return checkpoint_callback

    @staticmethod
    def get_pif():
        PIF_DATA_DIR = "/home/hiro819/projects/CrossPromptSAS/data/prompt_info.json"
        pif = json.load(open(PIF_DATA_DIR))
        return pif