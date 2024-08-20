# Reducing_cost

This repository includes scrpits for training BERT models for Reducing the cost: Cross-Prompt Pre-finetuning for Short Answer Scoring. 
There are three function in the main script.
1. **Training**: Fine-tuning a BERT model for specific prompt based on a given configuration.
2. **Evaluation**: Evaluating the fine-tuned model on a test dataset
3. **Zero-shot Evaluation**: Evaluating the model without further fine-tuning.

## Installation

> pip install -r requirement.txt

##Usage
### Training
To train a BERT model for a specific prompt with a conf file:
> python main.py train --config_path <path_to_config>

### Evaluation
To evaluate the fine-tuned model on a test dataset from a specific prompt:
>python main.py eval --config_path <path_to_config> [--test_path <path_to_test_data>] [--save_path <path_to_save_results>] [--prompt <prompt_id>] [--item <item_id>]

### Zero-shot evaluation
>python main.py eval_zero --config_path <path_to_config> --save_path <path_to_save_results>

### Data set
The dataset is available for academic use through the following link: https://www.nii.ac.jp/dsc/idr/rdata/RIKEN-SAA/
To use this scripts, you need to convert the json file to tsv file with three columns: answer, criteria and score.


