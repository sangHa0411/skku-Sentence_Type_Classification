import re
import numpy as np
from datasets import load_metric
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self, label_names, label_dict) :
        self.label_names = label_names
        self.label_dict = label_dict
        self.decoder = {i : l for l, i in label_dict.items()}
        self.tags = ['type', 'polarity', 'time', 'certainty']

    def compute_metrics(self, pred: EvalPrediction):
        labels = pred.label_ids
        predictions = pred.predictions
        
        metric = {}

        eval_size = len(labels)
        pred_args = predictions.argmax(-1)

        # 각 Label과 Prediction을 문자열로 변환
        labels_string = [self.decoder[l] for l in labels]
        predictions_string = [self.decoder[p] for p in pred_args]

        # 유형별로 정리, Accuracy 계산
        type_labels = [l.split('-')[0] for l in labels_string]
        type_preds = [p.split('-')[0] for p in predictions_string]
        type_acc = accuracy_score(type_labels, type_preds)
        metric['type-acc'] = type_acc

        # 극성별로 정리, Accuracy 계산
        polarity_labels = [l.split('-')[1] for l in labels_string]
        polarity_preds = [p.split('-')[1] for p in predictions_string]
        polarity_acc = accuracy_score(polarity_labels, polarity_preds)
        metric['polarity-acc'] = polarity_acc

        # 시제별로 정리, Accuracy 계산
        time_labels = [l.split('-')[2] for l in labels_string]
        time_preds = [p.split('-')[2] for p in predictions_string]
        time_acc = accuracy_score(time_labels, time_preds)
        metric['time-acc'] = time_acc

        # 확실성별로 정리, Accuracy 계산
        certainty_labels = [l.split('-')[3] for l in labels_string]
        certainty_preds = [p.split('-')[3] for p in predictions_string]
        certainty_acc = accuracy_score(certainty_labels, certainty_preds)
        metric['certainty-acc'] = certainty_acc
        
        # 전체적인 F1 Score 계산
        decoded_pred_vectors = []
        decoded_label_vectors = []
        for p, l  in zip(pred_args, labels) :
            pred_vector = [0] * len(self.label_names)
            pred_vector[p] = 1

            label_vector = [0] * len(self.label_names)
            label_vector[l] = 1

            decoded_pred_vectors.append(pred_vector)
            decoded_label_vectors.append(label_vector)

        # 구체적인 분석을 recall, precision을 같이 계산 
        f1 = f1_score(decoded_label_vectors, decoded_pred_vectors, average='weighted')
        recall = recall_score(decoded_label_vectors, decoded_pred_vectors, average='weighted')
        precision = precision_score(decoded_label_vectors, decoded_pred_vectors, average='weighted')

        metric['f1'] = f1
        metric['recall'] = recall
        metric['precision'] = precision

        return metric
