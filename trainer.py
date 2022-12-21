
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import FocalLoss
from transformers import Trainer
from typing import Dict, List, Any, Union, Tuple, Optional
from transformers.trainer_pt_utils import nested_detach

class Trainer(Trainer) :

    def __init__(self, *args, loss_fn, rdrop_flag, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.rdrop_flag = rdrop_flag

    # KL-Divergence Loss 계산하는 코드
    def get_kl_loss(self, loss_fn, logits_1, logits_2, alpha=1) :
        loss_kl_1 = loss_fn(F.log_softmax(logits_1, dim=-1), F.softmax(logits_2, dim=-1))
        loss_kl_2 = loss_fn(F.log_softmax(logits_2, dim=-1), F.softmax(logits_1, dim=-1))
        return alpha * (loss_kl_1 + loss_kl_2) / 2

    # 최적화 부분 : R-Drop을 이용해서 Loss를 계산한다. (Train 할 때만)
    def compute_loss(self, model, inputs):
        num_labels = model.config.num_labels

        if self.rdrop_flag :
            input_names = inputs.keys()
            batch_size = inputs['input_ids'].shape[0]
            labels = inputs.pop('labels')

            # 같은 입력을 2개 concat한다.
            for input_name in input_names :
                batch = inputs[input_name]
                inputs[input_name] = torch.cat([batch, batch], dim=0)

            # 출력을 구한다.
            outputs = model(**inputs)

            # 앞서 concat 한 부분을 나눈다.
            batch_logits_1 = outputs.logits[:batch_size, :]
            batch_logits_2 = outputs.logits[batch_size:, :]

            # 각 부분에 대해서 cross entropy loss의 평균을 구한다.            
            loss_fct_1 = self.loss_fn
            loss_nll = (
                loss_fct_1(batch_logits_1.view(-1, num_labels), labels.view(-1)) + \
                loss_fct_1(batch_logits_2.view(-1, num_labels), labels.view(-1))
            ) / 2

            # 두 부분의 logit 분포 차이가 적어질 수 있게 KL-Divergence Loss 구한다.
            loss_fct_2 = nn.KLDivLoss(reduction='batchmean')
            loss_kl = self.get_kl_loss(loss_fct_2, batch_logits_1, batch_logits_2)

            loss = loss_nll + loss_kl
        else :
            loss = self.compute_eval_loss(model, inputs)

        return loss

    # Validation 할 때 Loss를 구하기 위한 코드
    def compute_eval_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    # Huggingface Validation & Prediction Code
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with torch.no_grad():               
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_eval_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

            if prediction_loss_only:
                return (loss, None, None)

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

            return (loss, logits, labels)