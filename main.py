import pathlib
from os import path

import fire
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.SasTrainer import BertFineTuner
from src.evaluate import evaluate
from src.strcuture import Config
from src.util import Util

from loguru import logger as lloger


class Main:
    def __init__(self, ):
        pass

    def train(self, config_path):

        config = Config.load(pathlib.Path(config_path))
        seed_everything(config.seed, workers=True)

        # device = "gpu" if torch.cuda.is_available() else 'cpu'

        # config.model_dir = path.join(config.model_dir,save_suffix)

        checkpoint_callback = Util.get_callbacks(model_dir=config.model_dir,adaptive_pretrain=config.adaptive_pretrain)

        model = BertFineTuner(config, lloger)

        if config.use_wandb:
            logger = WandbLogger(name=config.wandb_name,project=config.wandb_project, config=config)
        else:
            logger = True

        train_params = dict(
            accumulate_grad_batches=config.gradient_accumulation_steps,
            max_epochs=config.num_train_epochs,
            precision=32,
            gradient_clip_val=config.max_grad_norm,
            enable_checkpointing=False,
            logger=logger,
            devices=config.n_gpu,
            accelerator="cuda",
            deterministic=True,
            num_sanity_val_steps=-1,
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # if not config.adaptive_pretrain:
        #     model.load_from_checkpoint(checkpoint_callback.best_model_path,config=config)

        if config.adaptive_pretrain:
            #save parameters of the last epoch
            model.tokenizer.save_pretrained(config.model_dir)
            model.model.save_pretrained(config.model_dir)
        else:
            #save best parameter on dev set
            model.tokenizer.save_pretrained(config.model_dir)
            model.best_model.save_pretrained(config.model_dir)

    def eval(self, config_path, test_path = None, save_path=None, prompt=None, item=None):

        config = Config.load(pathlib.Path(config_path))
        if test_path is not None:
            config.test_path = test_path
        if save_path is not None:
            config.save_path = save_path
        if prompt is not None:
            config.prompt = prompt
            config.item = item
        evaluate(config, logger=lloger, save_path = save_path, output_logits=True)

    def eval_zero(self, config_path, save_path):
        config = Config.load(pathlib.Path(config_path))
        evaluate(config, config.item, save_path=save_path, output_logits=True)




if __name__ == "__main__":
    fire.Fire(Main)
    pass
