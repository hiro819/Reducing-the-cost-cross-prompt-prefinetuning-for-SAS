import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from os import path
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from src.dataset import BertDataset
from src.util import Util


def evaluate(config, logger, item = None, save_path = None, output_logits = False):
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     "/content/checkpoints",
    #     monitor="val_loss", mode="min", save_top_k=1
    # )
    # トークナイザー（SentencePiece）

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name_or_path, is_fast=True)

    # # 学習済みモデル
    # trained_model = BertFineTuner(config)
    # checkpoint_path = glob.glob(path.join(config.model_dir,'checkpoint','*.ckpt'))[0]
    #
    # trained_model  = BertFineTuner.load_from_checkpoint(checkpoint_path, config=config)
    # print(f'load from {checkpoint_path}')

    trained_model = BertForSequenceClassification.from_pretrained(config.model_dir)
    print(f'Load from {config.model_dir}')

    # GPUの利用有無
    USE_GPU = torch.cuda.is_available()
    print(USE_GPU)
    if USE_GPU:
        trained_model.cuda()



    import textwrap
    from tqdm.auto import tqdm
    from sklearn import metrics

    # テストデータの読み込み
    test_dataset = BertDataset(tokenizer, config.test_path,
                              config=config)

    test_loader = DataLoader(test_dataset, batch_size=12, num_workers=4)

    trained_model.eval()

    outputs = []
    confidences = []
    targets = []
    logit_scores = []

    for batch in tqdm(test_loader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        outs = trained_model(input_ids=input_ids,
            attention_mask=input_mask)

        # dec = [tokenizer.decode(ids, skip_special_tokens=True,
        #                         clean_up_tokenization_spaces=False)
        #             for ids in outs.sequences]

        logits = outs.logits.squeeze()
        preds = Util.convert_to_original_score(logits,config.max_score)

        # conf = logits.max(dim=-1)


        target = Util.convert_to_original_score(batch['target_label'], config.max_score)
        # dec = map(lambda x:x.replace("点", ""),dec)
        outputs.extend(preds)
        # confidences.extend(conf)
        targets.extend(target)
        logit_scores.extend(logits.tolist())

    targets = list(map(int,targets))
    outputs = list(map(int, outputs))
    QWK = cohen_kappa_score(targets, outputs, weights='quadratic')
    logger.info(QWK)
    # print(min(confidences), max(confidences))
    metrics = []
    values = {}
    values['Metric'] = list()
    values['Value'] = list()
    values['Metric'].append("QWK")
    values['Value'].append(QWK)
    mse, rmse = calc_mse(targets,outputs, config.max_score)
    values['Metric'].append('MSE')
    values['Value'].append(mse)
    values['Metric'].append('RMSE')
    values['Value'].append(rmse)

    res_dict = {'target': targets, 'output': outputs}
    if output_logits:
        res_dict['logit'] = logit_scores
    res = pd.DataFrame().from_dict(res_dict)
    vl = pd.DataFrame().from_dict(values)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        res.to_csv(path.join(save_path, f"output.{config.prompt}.{config.item}.csv"))
        vl.to_csv(path.join(save_path, f"metric.{config.prompt}.{config.item}.csv"))

    else:
        os.makedirs(path.join(config.model_dir, f"result"), exist_ok=True)
        res.to_csv(path.join(config.model_dir,f"result/res.{config.prompt}.{item}.csv"))
        vl.to_csv(path.join(config.model_dir, f"result/metric.{config.prompt}.{config.item}.csv"))

    # conf_metrics, conf_values = conf_based_rmse(res, config.max_score)
    # metrics.extend(conf_metrics)
    # values.extend(conf_values)
    # eval_df = pd.DataFrame({'metrics': metrics, 'values': values})
    # eval_df.to_csv(path.join(config.model_dir,f"result/eval.{config.prompt_name}.{item}.csv"))

def calc_mse(targets, outputs, max_score):
    targets = np.array(targets) / max_score
    outputs = np.array(outputs) / max_score
    mse = np.mean((targets - outputs) ** 2)
    rmse = np.sqrt(mse)

    return mse, rmse


