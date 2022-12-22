import os
import random
import torch
import importlib
import numpy as np
import pandas as pd
import collections
import multiprocessing
import warnings
from tqdm import tqdm
from datasets import Dataset
from utils.encoder import Encoder
from utils.metrics import Metrics
from utils.encoder import Encoder
from utils.seperator import Seperator
from datasets import Dataset

from arguments import (
    DataTrainingArguments, 
    TrainingArguments, 
    InferenceArguments
)

from transformers.trainer_utils import EvalPrediction
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
    Trainer,
)

ENSEMBLE_SIZE = 1
MODEL_NAMES = [
    'RobertaSpecialTokenForSequenceClassification',
    'T5EncoderForSequenceClassification',
    'RobertaSpecialTokenForSequenceClassification',
]

MODEL_PATHS = [
    './exps/model6',
    './exps/model8',
    './exps/model5',
]


def main():

    warnings.filterwarnings(action='ignore')

    parser = HfArgumentParser(
        (DataTrainingArguments, TrainingArguments, InferenceArguments)
    )
    data_args, training_args, inference_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)
    
    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)
    training_args.dataloader_num_workers = num_proc

    # -- Loading datasets
    print("\nLoad datasets")
    file_path = os.path.join(data_args.data_dir, data_args.train_data_file)
    train_df = pd.read_csv(file_path)
    train_df = train_df.drop_duplicates(subset=['문장'])

    # -- Label Tags
    label_names = list(train_df['label'].unique())
    label_dict = {l:i for i, l in enumerate(label_names)}

    # -- Seperating validation datasets
    seperator = Seperator(validation_ratio=data_args.validation_ratio)
    datasets = seperator(train_df)
    validation_dataset = datasets['validation']

    labels = np.array([label_dict[l] for l in validation_dataset['label']])
    validation_dataset = validation_dataset.remove_columns(['label'])

    # -- Remove label compumns
    print(validation_dataset)

    predictions_list = []
    for i in range(ENSEMBLE_SIZE) :
        model_name = MODEL_NAMES[i]
        model_path = MODEL_PATHS[i]

        # -- Encoding datasets
        print("\nEncoding dataset")   
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        encoder = Encoder(tokenizer, data_args.max_length, label_dict=None, train_flag=False)
        sub_dataset = validation_dataset.map(encoder, batched=True, num_proc=num_proc)
        sub_dataset = sub_dataset.remove_columns(['ID', '문장', '유형', '극성', '시제', '확실성', '__index_level_0__'])
        print(sub_dataset)
    
        # -- Loading Config
        config = AutoConfig.from_pretrained(model_path)

        if 'roberta' in model_name.lower() :
            model_category = importlib.import_module('models.roberta')
        elif 'electra' in model_name.lower() :
            model_category = importlib.import_module('models.electra')
        elif 't5' in model_name.lower() :
            model_category = importlib.import_module('models.t5')

        # -- Loading Model
        model_class = getattr(model_category, model_name)
        model = model_class.from_pretrained(model_path, config=config)

        # -- DataCollator
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, 
            padding=True,
            max_length=data_args.max_length   
        )

        # -- Trainer
        trainer = Trainer(
            model,
            training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # -- Prediction
        prediction_logits = trainer.predict(test_dataset=sub_dataset).predictions
        prediction_logits = np.exp(prediction_logits)
        prediction_probs = prediction_logits / np.expand_dims(np.sum(prediction_logits, axis=-1), -1)

        predictions_list.append(prediction_probs)

    # -- Soft Voting
    predictions = np.mean(predictions_list, axis=0)
    eval_preds = EvalPrediction(predictions=predictions, label_ids=labels)

    # -- Metrics
    metrics = Metrics(label_names, label_dict)
    eval_logs = metrics.compute_metrics(eval_preds)

    print(eval_logs)

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