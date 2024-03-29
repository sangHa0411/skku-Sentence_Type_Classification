import os
import wandb
import random
import importlib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import multiprocessing
from dotenv import load_dotenv
from trainer import Trainer
from utils.metrics import Metrics
from utils.encoder import Encoder
from utils.seperator import Seperator
from utils.augmentation import Augmentation
from datasets import Dataset, DatasetDict
import warnings

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
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    
    warnings.filterwarnings(action='ignore')

    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Arguments
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    training_args.dataloader_num_workers = num_proc
    
    # -- Seed 
    seed_everything(training_args.seed)

    # -- Loading datasets
    print("\nLoad datasets")
    file_path = os.path.join(data_args.data_dir, data_args.train_data_file)
    train_df = pd.read_csv(file_path)
    train_df = train_df.drop_duplicates(subset=['문장'])

    label_names = list(train_df['label'].unique())
    label_dict = {l:i for i, l in enumerate(label_names)}

    # -- Preprocessing datasets
    print("\nPreprocessing datasets")   
    if training_args.do_eval :
        file_path = os.path.join(data_args.data_dir, data_args.validation_data_file)
        validation_df = pd.read_csv(file_path)
        validation_df = validation_df.drop_duplicates(subset=['문장'])

        train_dataset = Dataset.from_pandas(train_df)
        validation_dataset = Dataset.from_pandas(validation_df)

        dataset = DatasetDict({'train' : train_dataset, 'validation' : validation_dataset})
        print(dataset)

    else :
        dataset = Dataset.from_pandas(train_df)
        print(dataset)

    # -- Encoding datasets
    print("\nEncoding datasets")
    
    # -- Load tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # -- Encoding datasets
    print("\nEncoding datasets")
    encoder = Encoder(tokenizer, data_args.max_length, label_dict)
    if training_args.do_eval :
        # augmentator = Augmentation(
        #     tokenizer=tokenizer,
        #     max_num=data_args.maximum_size, 
        #     min_num=data_args.minimum_size, 
        #     eval_flag=True
        # )

        dataset = dataset.map(encoder, batched=True, num_proc=num_proc)
        # train_dataset = datasets['train']
        # eval_dataset = datasets['validation']

        # train_dataset = augmentator(train_dataset)
        # datasets = DatasetDict({'train' : train_dataset, 'validation' : eval_dataset})
        dataset = dataset.remove_columns(['문장', '유형', '극성', '시제', '확실성', 'label', '__index_level_0__'])
        print(dataset)
    else :
        augmentator = Augmentation(
            tokenizer=tokenizer,
            max_num=data_args.maximum_size, 
            min_num=data_args.minimum_size, 
            eval_flag=False
        )
        dataset = dataset.map(encoder, batched=True, num_proc=num_proc)
        dataset = augmentator(dataset)
        dataset = dataset.remove_columns(['문장', '유형', '극성', '시제', '확실성', 'label', '__index_level_0__'])
        print(dataset)

    # -- Loading config & Model
    print("\nLoad Model")
    config = AutoConfig.from_pretrained(model_args.PLM)
    ## -- Labels
    config.num_labels = len(label_names)
    ## -- Speical Token ids
    config.cls_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.eos_token_id

    ## -- Model Category
    model_category = importlib.import_module('models.roberta')
    ## -- Model Class
    model_class = getattr(model_category, model_args.model_name)
    model = model_class.from_pretrained(model_args.PLM, config=config)

    # -- DataCollator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True,
        max_length=data_args.max_length   
    )

    # -- Metrics
    metrics = Metrics(label_names, label_dict)
    compute_metrics = metrics.compute_metrics

    load_dotenv(dotenv_path=logging_args.dotenv_path)

    # -- Wandb Setting
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
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else :
        trainer = Trainer(
            model,
            training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    
    # -- Training
    print('\nTraining')
    trainer.train()
    
    if training_args.do_eval :
        # -- Evaluation
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
