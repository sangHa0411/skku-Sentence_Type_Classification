
import numpy as np
from sklearn.metrics import f1_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self,) :
        self.categories = ['type', 'polarity', 'time', 'certainty']

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

        total_acc = 0.0
        for i in range(eval_size) :
            if labels[0][i] == pred_args1[i] and \
                labels[1][i] == pred_args2[i] and \
                labels[2][i] == pred_args3[i] and \
                labels[3][i] == pred_args4[i] :
                total_acc += 1.0
                
        total_acc /= eval_size
        metric['total_acc'] = total_acc
        return metric