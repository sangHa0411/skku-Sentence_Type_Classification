import os
import torch
import random
import wandb
import numpy as np
import pandas as pd
import importlib
import multiprocessing
from dotenv import load_dotenv
from datasets import DatasetDict
# from trainer import Trainer
from utils.metrics import Metrics
from utils.encoder import Encoder
from utils.seperator import Seperator
from utils.augmentation import Augmentation
from datasets import Dataset, DatasetDict

from arguments import (
    ModelArguments, 
    DataTrainingArguments, 
    TrainingArguments, 
    LoggingArguments
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
    Trainer,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    
    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Arguments
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    training_args.dataloader_num_workers = num_proc
    seed_everything(training_args.seed)

    # -- Loading datasets
    print("\nLoad datasets")
    file_path = os.path.join(data_args.data_dir, data_args.train_data_file)
    df = pd.read_csv(file_path)

    # -- Preprocessing datasets
    print("\nPreprocessing datasets")   
    if training_args.do_eval :
        seperator = Seperator(validation_ratio=data_args.validation_ratio)
        datasets = seperator(df)
        print(datasets)
    else :
        dataset = Dataset.from_pandas(df)
        print(dataset)

    # -- Encoding datasets
    print("\nEncoding datasets")
    label_dict = {
        '유형' : {'사실형': 0, '추론형': 1, '대화형': 2, '예측형' : 3},
        '극성' : {'긍정': 0, '부정': 1, '미정': 2},
        '시제' : {'과거': 0, '현재': 1, '미래': 2},
        '확실성' : {'확실': 0, '불확실': 1},
    }
    
    # -- Load tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    augmentator = Augmentation(max_num=data_args.maximum_size, min_num=data_args.minimum_size)

    # -- Encoding datasets
    print("\nEncoding datasets")
    encoder = Encoder(tokenizer, data_args.max_length, label_dict)
    if training_args.do_eval :
        datasets = datasets.map(encoder, batched=True, num_proc=num_proc)
        train_dataset = datasets['train']
        eval_dataset = datasets['validation']

        train_dataset = augmentator(train_dataset)
        datasets = DatasetDict({'train' : train_dataset, 'validation' : eval_dataset})
        datasets = datasets.remove_columns(['ID', '문장', '유형', '극성', '시제', '확실성', 'label'])
        print(datasets)
    else :
        dataset = dataset.map(encoder, batched=True, num_proc=num_proc)
        dataset = augmentator(dataset)
        dataset = dataset.remove_columns(['ID', '문장', '유형', '극성', '시제', '확실성', 'label'])
        print(dataset)

    # -- Loading config & Model
    print("\nLoad Model")
    config = AutoConfig.from_pretrained(model_args.PLM)
    config.category1_num_labels = len(label_dict['유형'])
    config.category2_num_labels = len(label_dict['극성'])
    config.category3_num_labels = len(label_dict['시제'])
    config.category4_num_labels = len(label_dict['확실성'])
    
    if 'roberta' in model_args.PLM :
        model_category = importlib.import_module('models.roberta')
    elif 'electra' in model_args.PLM :
        model_category = importlib.import_module('models.electra')
    elif 't5' in model_args.PLM :
        model_category = importlib.import_module('models.t5')
    else :
        raise NotImplementedError('Not implemented model type')

    model_class = getattr(model_category, model_args.model_name)
    model = model_class.from_pretrained(model_args.PLM, config=config)

    # -- DataCollator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True,
        max_length=data_args.max_length   
    )

    # -- Metrics
    metrics = Metrics()
    compute_metrics = metrics.compute_metrics

    # -- Output Directory
    load_dotenv(dotenv_path=logging_args.dotenv_path)

    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    args = training_args
    if args.max_steps == -1 :
        wandb_name = f'EP:{args.num_train_epochs}_BS:{args.per_device_train_batch_size}_LR:{args.learning_rate}_WD:{args.weight_decay}_WR:{args.warmup_ratio}'
    else :
        wandb_name = f'MS:{args.max_steps}_BS:{args.per_device_train_batch_size}_LR:{args.learning_rate}_WD:{args.weight_decay}_WR:{args.warmup_ratio}'

    wandb.init(
        entity='sangha0411',
        project=logging_args.project_name, 
        name=wandb_name,
        group=logging_args.group_name
    )
    wandb.config.update(training_args)

    # -- Trainer
    if training_args.do_eval :
        trainer = Trainer(
            model,
            training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    else :
        trainer = Trainer(
            model,
            training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    print('\nTraining')
    trainer.train()

    if training_args.do_eval :
        print('\nEvaluating')
        eval_log = trainer.evaluate()
        print(eval_log)

    if training_args.do_eval == False :
        trainer.save_model(training_args.output_dir)
    wandb.finish()


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()