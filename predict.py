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
    file_path = os.path.join(data_args.data_dir, inference_args.test_data_file)
    df = pd.read_csv(file_path)

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
    prediction_logits = trainer.predict(test_dataset=dataset)[0]

    pred_args1 = prediction_logits[0].argmax(-1)
    pred_args2 = prediction_logits[1].argmax(-1)
    pred_args3 = prediction_logits[2].argmax(-1)
    pred_args4 = prediction_logits[3].argmax(-1)

    # -- Postprocess
    label_dict = {
        '유형' : {0 : '사실형', 1 : '추론형', 2 : '대화형', 3 : '예측형'},
        '극성' : {0 : '긍정', 1 : '부정', 2 : '미정'},
        '시제' : {0 : '과거', 1 : '현재', 2 : '미래'},
        '확실성' : {0 : '확실', 1 : '불확실'},
    }

    labels = []
    test_size = len(dataset['input_ids'])
    for i in tqdm(range(test_size)) :
        decoded_string = label_dict['유형'][pred_args1[i]] + '-' + \
            label_dict['극성'][pred_args2[i]] + '-' + \
            label_dict['시제'][pred_args3[i]] + '-' + \
            label_dict['확실성'][pred_args4[i]]

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