import re
import numpy as np
from datasets import load_metric
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self, label_names, label2index) :
        self.categories = ['type', 'polarity', 'time', 'certainty']
        self.label_dict = {
            '유형' : {0 : '사실형', 1 : '추론형', 2 : '대화형', 3 : '예측형'},
            '극성' : {0 : '긍정', 1 : '부정', 2 : '미정'},
            '시제' : {0 : '과거', 1 : '현재', 2 : '미래'},
            '확실성' : {0 : '확실', 1 : '불확실'},
        }
        self.label_names = label_names
        self.label2index = label2index


    def calculate_f1(self, labels, preds, size) :
        label_vectors = []
        pred_vectors = []

        for l, p in zip(labels, preds) :
            l_vector = [0]*size
            l_vector[l] = 1
            label_vectors.append(l_vector)

            p_vector = [0]*size
            p_vector[p] = 1
            pred_vectors.append(p_vector)

        f1 = f1_score(label_vectors, pred_vectors, average='weighted')
        return f1


    def compute_metrics(self, pred: EvalPrediction):
        labels = pred.label_ids
        predictions = pred.predictions[4:]
        
        metric = {}

        eval_size = len(labels[0])
        pred_args1 = predictions[0].argmax(-1)
        pred_args2 = predictions[1].argmax(-1)
        pred_args3 = predictions[2].argmax(-1)
        pred_args4 = predictions[3].argmax(-1)

        metric[self.categories[0] + '-f1'] = self.calculate_f1(labels[0], pred_args1, 4)
        metric[self.categories[1] + '-f1'] = self.calculate_f1(labels[1], pred_args2, 3)
        metric[self.categories[2] + '-f1'] = self.calculate_f1(labels[2], pred_args3, 3)
        metric[self.categories[3] + '-f1'] = self.calculate_f1(labels[3], pred_args4, 2)

        decoded_predictions = []
        decoded_labels = []

        total_acc = 0.0
        for i in range(eval_size) :

            label = [
                self.label_dict['유형'][labels[0][i]], 
                self.label_dict['극성'][labels[1][i]], 
                self.label_dict['시제'][labels[2][i]], 
                self.label_dict['확실성'][labels[3][i]]
            ]

            pred = [
                self.label_dict['유형'][pred_args1[i]], 
                self.label_dict['극성'][pred_args2[i]], 
                self.label_dict['시제'][pred_args3[i]], 
                self.label_dict['확실성'][pred_args4[i]]
            ]

            label = '-'.join(label)
            decoded_labels.append(label)

            pred = '-'.join(pred)
            decoded_predictions.append(pred)

        decoded_prediction_vectors = []
        decoded_label_vectors = []
        for p, l  in zip(decoded_predictions, decoded_labels) :
            pred_vector = [0] * len(self.label_names)
            if p in self.label_names :
                p_id = self.label_names.index(p)
                pred_vector[p_id] = 1

            label_vector = [0] * len(self.label_names)
            l_id = self.label_names.index(l)
            label_vector[l_id] = 1

            decoded_prediction_vectors.append(pred_vector)
            decoded_label_vectors.append(label_vector)

        f1 = f1_score(decoded_label_vectors, decoded_prediction_vectors, average='weighted')
        metric['total-f1'] = f1
 
        return metric
