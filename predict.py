import os
import importlib
import pandas as pd
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


def main():
    parser = HfArgumentParser(
        (DataTrainingArguments, TrainingArguments, InferenceArguments)
    )
    data_args, training_args, inference_args = parser.parse_args_into_dataclasses()

    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)
    training_args.dataloader_num_workers = num_proc

    # -- Loading datasets
    print("\nLoad datasets")
    file_path = os.path.join(data_args.data_dir, data_args.train_data_file)
    train_df = pd.read_csv(file_path)
    train_df = train_df.drop_duplicates(subset=['문장'])

    label_names = list(train_df['label'].unique())
    label_dict = {i:l for i, l in enumerate(label_names)}

    file_path = os.path.join(data_args.data_dir, inference_args.test_data_file)
    df = pd.read_csv(file_path)
    test_size = len(df)

    label_names = list(df['label'].unique())
    label_dict = {i:l for i, l in enumerate(label_names)}

    # -- Parsing datasets
    print("\nParse dataset")   
    dataset = Dataset.from_pandas(df)
    print(dataset)

    # -- Encoding datasets
    print("\nEncoding dataset")   
    model_name = inference_args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    encoder = Encoder(tokenizer, data_args.max_length, label_dict=None, train_flag=False)
    dataset = dataset.map(encoder, batched=True, num_proc=num_proc)
    print(dataset)

    # -- Loading Model
    config = AutoConfig.from_pretrained(model_name)
    if 'roberta' in inference_args.model_type :
        model_category = importlib.import_module('models.roberta')
    elif 'electra' in inference_args.model_type :
        model_category = importlib.import_module('models.electra')
    elif 't5' in inference_args.model_type :
        model_category = importlib.import_module('models.t5')

    model_class = getattr(model_category, inference_args.model_name)
    model = model_class.from_pretrained(model_name, config=config)

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
    prediction_logits = trainer.predict(test_dataset=dataset)
    pred_args = prediction_logits.predictions.argmax(-1)

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