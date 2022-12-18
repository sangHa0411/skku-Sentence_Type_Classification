
import numpy as np
from datasets import load_metric
from sklearn.metrics import f1_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self,) :
        self.metric = load_metric("f1")
        self.categories = ['type', 'polarity', 'time', 'certainty']
        self.label_dict = {
            '유형' : {0 : '사실형', 1 : '추론형', 2 : '대화형', 3 : '예측형'},
            '극성' : {0 : '긍정', 1 : '부정', 2 : '미정'},
            '시제' : {0 : '과거', 1 : '현재', 2 : '미래'},
            '확실성' : {0 : '확실', 1 : '불확실'},
        }
        self.mapping = self.get_mapping(self.label_dict)

    def compute_metrics(self, pred: EvalPrediction):
        labels = pred.label_ids
        predictions = pred.predictions[4:]
        
        metric = {}
        for i in range(len(self.categories)) :
            category_labels = labels[i]
            category_predictions = predictions[i]
            category_predictions = category_predictions.argmax(-1)

            micro_f1 = f1_score(category_labels, category_predictions, average="micro")
            macro_f1 = f1_score(category_labels, category_predictions, average="macro")

            metric[self.categories[i] + '-micro_f1'] = micro_f1
            metric[self.categories[i] + '-macro_f1'] = macro_f1


        eval_size = len(labels[0])
        pred_args1 = predictions[0].argmax(-1)
        pred_args2 = predictions[1].argmax(-1)
        pred_args3 = predictions[2].argmax(-1)
        pred_args4 = predictions[3].argmax(-1)

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

        decoded_labels = [self.mapping[l] for l in decoded_labels]
        decoded_predictions = [self.mapping[l] for l in decoded_predictions]

        micro_f1 = self.metric.compute(predictions=decoded_predictions, references=decoded_labels, average='micro')
        macro_f1 = self.metric.compute(predictions=decoded_predictions, references=decoded_labels, average='macro')

        metric['micro_f1'] = micro_f1['f1']
        metric['macro_f1'] = macro_f1['f1']
        return metric


    def get_mapping(self, label_dict) :

        mapping = {}

        tag1 = self.label_dict['유형']
        tag2 = self.label_dict['극성']
        tag3 = self.label_dict['시제']
        tag4 = self.label_dict['확실성']

        index = 0
        for i in range(len(tag1)) :
            for j in range(len(tag2)) :
                for k in range(len(tag3)) :
                    for l in range(len(tag4)) :

                        tag_string = '-'.join([tag1[i], tag2[j], tag3[k], tag4[l]])
                        mapping[tag_string] = index
                        index += 1

        return mapping