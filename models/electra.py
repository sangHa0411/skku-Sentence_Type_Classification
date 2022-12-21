
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from models.loss import FocalLoss
from typing import Optional, Tuple, Union
from transformers import ElectraPreTrainedModel, ElectraModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.activations import get_activation


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, classifier_dropout, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = ClassificationHead(config.hidden_size, config.hidden_dropout_prob, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

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
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class ElectraVStackSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = ClassificationHead(config.hidden_size, config.hidden_dropout_prob, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_hidden_states = True
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
        hidden_states = outputs[1]

        sequence_output = hidden_states[-1] * 0.6 + \
            hidden_states[-2] * 0.3 + \
            hidden_states[-3] * 0.1

        sequence_output = sequence_output[:, 0, :]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs.hidden_states = None
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
