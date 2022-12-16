

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers import ElectraPreTrainedModel, ElectraModel
from models.output import SequenceClassifierOutput
from transformers.activations import get_activation

class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x) 
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.electra = ElectraModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, config.category1_num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, config.category2_num_labels)
        self.classifier3 = nn.Linear(config.hidden_size, config.category3_num_labels)
        self.classifier4 = nn.Linear(config.hidden_size, config.category4_num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        labels3: Optional[torch.Tensor] = None,
        labels4: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0][:, 0, :]

        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)
        logits3 = self.classifier3(sequence_output)
        logits4 = self.classifier4(sequence_output)

        loss = None
        loss1, loss2, loss3, loss4 = None, None, None, None
        if labels1 is not None and labels2 is not None and labels3 is not None and labels4 is not None:
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, self.config.category1_num_labels), labels1.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.config.category2_num_labels), labels2.view(-1))
            loss3 = loss_fct(logits3.view(-1, self.config.category3_num_labels), labels3.view(-1))
            loss4 = loss_fct(logits4.view(-1, self.config.category4_num_labels), labels4.view(-1))

            loss = loss1 + loss2 + loss3 + loss4

        if not return_dict:
            output = (logits1, logits2, logits3, logits4, ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1, logits2, logits3, logits4, ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            category1_loss=loss1,
            category2_loss=loss2,
            category3_loss=loss3,
            category4_loss=loss4,
            category1_logits=logits1,
            category2_logits=logits2,
            category3_logits=logits3,
            category4_logits=logits4,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )