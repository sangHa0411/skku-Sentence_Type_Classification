import re
import numpy as np
from datasets import load_metric
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self, label_names, label_dict) :
        self.label_names = label_names
        self.label_dict = label_dict

    def compute_metrics(self, pred: EvalPrediction):
        labels = pred.label_ids
        predictions = pred.predictions
        
        metric = {}

        eval_size = len(labels)
        pred_args = predictions.argmax(-1)
       
        decoded_pred_vectors = []
        decoded_label_vectors = []
        for p, l  in zip(pred_args, labels) :
            pred_vector = [0] * len(self.label_names)
            pred_vector[p] = 1

            label_vector = [0] * len(self.label_names)
            label_vector[l] = 1

            decoded_pred_vectors.append(pred_vector)
            decoded_label_vectors.append(label_vector)

        f1 = f1_score(decoded_label_vectors, decoded_pred_vectors, average='weighted')
        recall = recall_score(decoded_label_vectors, decoded_pred_vectors, average='weighted')
        precision = precision_score(decoded_label_vectors, decoded_pred_vectors, average='weighted')

        metric['f1'] = f1
        metric['recall'] = recall
        metric['precision'] = precision

        return metric
