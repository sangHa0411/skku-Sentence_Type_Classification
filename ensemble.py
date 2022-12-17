import os
import importlib
import numpy as np
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

ENSEMBLE_SIZE = 3
MODEL_NAMES = [
    'RobertaBaseForSequenceClassification',
    'RobertaFocalForSequenceClassification',
    'T5EncoderArcFaceForSequenceClassification'
]
MODEL_PATHS = [
    './exps/model1/checkpoint-2000',
    './exps/model2/checkpoint-2000',
    './exps/model3/checkpoint-2000',
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

    # -- Loading datasets
    print("\nLoad datasets")
    file_path = os.path.join(data_args.data_dir, inference_args.test_data_file)
    df = pd.read_csv(file_path)

    # -- Parsing datasets
    print("\nParse dataset")   
    dataset = Dataset.from_pandas(df)
    print(dataset)

    logits1, logits2, logits3, logits4 = [], [], [], []
    
    for i in range(ENSEMBLE_SIZE) :
        model_name = MODEL_NAMES[i]
        model_path = MODEL_PATHS[i]

        # -- Encoding datasets
        print("\nEncoding dataset")   
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        encoder = Encoder(tokenizer, data_args.max_length, label_dict=None, train_flag=False)
        dataset = dataset.map(encoder, batched=True, num_proc=num_proc)
        print(dataset)
    
        # -- Loading Model
        config = AutoConfig.from_pretrained(model_path)

        if 'roberta' in model_name.lower() :
            model_category = importlib.import_module('models.roberta')
        elif 'electra' in model_name.lower() :
            model_category = importlib.import_module('models.electra')
        elif 't5' in model_name.lower() :
            model_category = importlib.import_module('models.t5')

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
        prediction_logits = trainer.predict(test_dataset=dataset)[0]

        # -- Softmax
        prediction_logits1 = np.exp(prediction_logits[0])
        predictions_probs1 = prediction_logits1 / np.expand_dims(np.sum(prediction_logits1, axis=-1), -1)

        prediction_logits2 = np.exp(prediction_logits[1])
        predictions_probs2 = prediction_logits2 / np.expand_dims(np.sum(prediction_logits2, axis=-1), -1)

        prediction_logits3 = np.exp(prediction_logits[2])
        predictions_probs3 = prediction_logits3 / np.expand_dims(np.sum(prediction_logits3, axis=-1), -1)

        prediction_logits4 = np.exp(prediction_logits[3])
        predictions_probs4 = prediction_logits4 / np.expand_dims(np.sum(prediction_logits4, axis=-1), -1)

        logits1.append(predictions_probs1)
        logits2.append(predictions_probs2)
        logits3.append(predictions_probs3)
        logits4.append(predictions_probs4)

    # -- Selecting arguments
    pred_args1 = np.mean(logits1, axis=0).argmax(-1)
    pred_args2 = np.mean(logits2, axis=0).argmax(-1)
    pred_args3 = np.mean(logits3, axis=0).argmax(-1)
    pred_args4 = np.mean(logits4, axis=0).argmax(-1)

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