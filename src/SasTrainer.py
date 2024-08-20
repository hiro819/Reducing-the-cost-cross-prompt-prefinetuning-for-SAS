from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.dataset import BertDataset
from torch.optim import  Adam

from transformers import (
    BertForSequenceClassification,
    BertTokenizer
)

import copy

import torch

from src.util import Util
from sklearn.metrics import cohen_kappa_score


from torch import nn

class BertFineTuner(pl.LightningModule):
    def __init__(self, config, llogger):
        super().__init__()
        self.config = config
        self.llogger = llogger

        #load pretrained bert model for regression
        # self.model = BertSAS(self.config, num_labels=1)
        self.model = BertForSequenceClassification.from_pretrained(config.model_name_or_path, num_labels=1)
        self.best_model = None

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name_or_path)

        #for calculate qwk
        self.preds = []
        self.targets =[]

        if config.adaptive_pretrain:
            self.best_mse = 1.0
            self.mse = nn.MSELoss()
        else:
            self.best_qwk = -1.0

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """loss_cal_for_train"""

        input_ids = batch["source_ids"].to(self.device)
        attention_mask = batch["source_mask"].to(self.device)
        labels = batch["target_label"].to(self.device)
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        return loss

    def _step_val(self, batch):
        """loss_cal_for_valid"""

        input_ids = batch["source_ids"].to(self.device)
        attention_mask = batch["source_mask"].to(self.device)
        labels = batch["target_label"].to(self.device)
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        pred = outputs[1].squeeze()

        #qwk計算用
        pred = Util.convert_to_original_score(pred,self.config.max_score)

        target = Util.convert_to_original_score(batch['target_label'], self.config.max_score)

        return loss, pred,target

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        self.llogger.info(f'train loss: {loss}')
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, pred,target = self._step_val(batch)
        self.log("val_loss", loss)
        self.preds.extend(pred)
        self.targets.extend(target)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        if self.config.adaptive_pretrain:
            preds = torch.cat(self.preds)
            targets = torch.cat(self.targets)
            mse = self.mse(preds,targets)
            self.log("val_epoch_mse", mse, prog_bar=True)

            if self.best_mse > mse:
                self.best_model = copy.deepcopy(self.model.to('cpu'))
                self.model.to('cuda')
                self.best_mse = mse

        else:
            # preds = torch.cat(self.preds
            # targets = torch.cat(self.targets).to('cpu')
            qwk = cohen_kappa_score(self.preds, self.targets)
            print(self.preds, self.targets)
            self.preds = []
            self.targets = []
            self.log("val_epoch_qwk", qwk, prog_bar=True)
            self.llogger.info(f"[Valid] {self.current_epoch} epoch, QWK: {qwk}")
            if self.best_qwk <= qwk:
                self.best_model = copy.deepcopy(self.model.to('cpu'))
                self.model.to('cuda')
                # self.model.save_pretrained(self.config.model_dir)
                self.best_qwk = qwk

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    # def test_epoch_end(self, outputs):
    #     """テスト完了処理"""
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     self.log("test_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        model = self.model
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters()
        #                     if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.hparams.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters()
        #                     if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)


        return optimizer

    def get_dataset(self, tokenizer, file_path, config):
        print(config)
        return BertDataset(
            tokenizer=tokenizer,
            config=config,
            file_path=file_path,
        )


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:

            train_dataset = self.get_dataset(tokenizer=self.tokenizer,file_path= self.config.train_path, config = self.config)

            self.train_dataset = train_dataset


            val_dataset = self.get_dataset(tokenizer=self.tokenizer, file_path= self.config.dev_path, config=self.config)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self.config.train_batch_size * max(1, self.config.n_gpu)))
                // self.config.gradient_accumulation_steps
                * float(self.config.num_train_epochs)
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.train_batch_size,
                          drop_last=False, shuffle=True, num_workers=8,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.eval_batch_size,
                          drop_last=False, shuffle=False,num_workers=4)