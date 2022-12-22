import os
import importlib
import numpy as np
import pandas as pd
import collections
import multiprocessing
from tqdm import tqdm
from datasets import Dataset
from utils.encoder import Encoder
from datasets import Dataset

from arguments import (
    DataTrainingArguments, 
    TrainingArguments, 
    InferenceArguments
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
    Trainer,
)

ENSEMBLE_SIZE = 5
MODEL_NAME = 'RobertaSpecialTokenForSequenceClassification'
MODEL_PATHS = [
    './exps/model1/checkpoint-2000',
    './exps/model2/checkpoint-2000',
    './exps/model3/checkpoint-2000',
    './exps/model4/checkpoint-2000',
    './exps/model5/checkpoint-2000',
]


def main():
    parser = HfArgumentParser(
        (DataTrainingArguments, TrainingArguments, InferenceArguments)
    )
    data_args, training_args, inference_args = parser.parse_args_into_dataclasses()

    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)
    training_args.dataloader_num_workers = num_proc

    # -- Loading train datasets
    print("\nLoad datasets")
    file_path = os.path.join(data_args.data_dir, data_args.train_data_file)
    train_df = pd.read_csv(file_path)
    train_df = train_df.drop_duplicates(subset=['문장'])

    # -- Label Tags
    label_names = list(train_df['label'].unique())
    label_dict = {i:l for i, l in enumerate(label_names)}
    
    # -- Loading test dataset
    file_path = os.path.join(data_args.data_dir, inference_args.test_data_file)
    df = pd.read_csv(file_path)

    # -- Parsing datasets
    print("\nParse dataset")   
    dataset = Dataset.from_pandas(df)
    print(dataset)

    test_size = len(dataset['ID'])

    predictions_list = []
    
    for i in range(ENSEMBLE_SIZE) :
        model_name = MODEL_NAME
        model_path = MODEL_PATHS[i]

        # -- Encoding datasets
        print("\nEncoding dataset")   
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        encoder = Encoder(tokenizer, data_args.max_length, label_dict=None, train_flag=False)
        sub_dataset = dataset.map(encoder, batched=True, num_proc=num_proc)
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
    pred_args = predictions.argmax(-1)

    # -- Postprocess
    labels = []
    for i in tqdm(range(test_size)) :
        decoded_string = label_dict[pred_args[i]]
        labels.append(decoded_string)

    # -- Submission
    submission_df = pd.read_csv(
        os.path.join(data_args.data_dir, inference_args.submission_data_file)
    )

    submission_df['label'] = labels
    submission_df.to_csv(
        os.path.join(inference_args.save_dir, inference_args.save_file_name), 
        index=False
    )

if __name__ == "__main__":
    main()