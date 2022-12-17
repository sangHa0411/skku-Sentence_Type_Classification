import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from models.loss import FocalLoss, ArcFace
from models.output import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

class T5EncoderBaseForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier1 = nn.Linear(config.d_model, config.category1_num_labels)
        self.classifier2 = nn.Linear(config.d_model, config.category2_num_labels)
        self.classifier3 = nn.Linear(config.d_model, config.category3_num_labels)
        self.classifier4 = nn.Linear(config.d_model, config.category4_num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels1:Optional[torch.FloatTensor] = None,
        labels2:Optional[torch.FloatTensor] = None,
        labels3:Optional[torch.FloatTensor] = None,
        labels4:Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        sequence_output = hidden_states[input_ids == self.config.eos_token_id]
        sequence_output = self.dropout(sequence_output)

        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)
        logits3 = self.classifier3(sequence_output)
        logits4 = self.classifier4(sequence_output)

        loss = None
        loss1, loss2, loss3, loss4 = None, None, None, None
        if labels1 is not None and labels2 is not None and labels3 is not None and labels4 is not None:
            
            loss_fct = nn.CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, self.config.category1_num_labels), labels1.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.config.category2_num_labels), labels2.view(-1))
            loss3 = loss_fct(logits3.view(-1, self.config.category3_num_labels), labels3.view(-1))
            loss4 = loss_fct(logits4.view(-1, self.config.category4_num_labels), labels4.view(-1))

            loss = (loss1 + loss2 + loss3 + loss4) / 4

        if not return_dict:
            output = (logits1, logits2, logits3, logits4, ) + encoder_outputs[2:]
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
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class T5EncoderFocalForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier1 = nn.Linear(config.d_model, config.category1_num_labels)
        self.classifier2 = nn.Linear(config.d_model, config.category2_num_labels)
        self.classifier3 = nn.Linear(config.d_model, config.category3_num_labels)
        self.classifier4 = nn.Linear(config.d_model, config.category4_num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels1:Optional[torch.FloatTensor] = None,
        labels2:Optional[torch.FloatTensor] = None,
        labels3:Optional[torch.FloatTensor] = None,
        labels4:Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        sequence_output = hidden_states[input_ids == self.config.eos_token_id]
        sequence_output = self.dropout(sequence_output)

        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)
        logits3 = self.classifier3(sequence_output)
        logits4 = self.classifier4(sequence_output)

        loss = None
        loss1, loss2, loss3, loss4 = None, None, None, None
        if labels1 is not None and labels2 is not None and labels3 is not None and labels4 is not None:
            
            loss_fct = FocalLoss(gamma=1, alpha=0.25)
            loss1 = loss_fct(logits1.view(-1, self.config.category1_num_labels), labels1.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.config.category2_num_labels), labels2.view(-1))
            loss3 = loss_fct(logits3.view(-1, self.config.category3_num_labels), labels3.view(-1))
            loss4 = loss_fct(logits4.view(-1, self.config.category4_num_labels), labels4.view(-1))

            loss = loss1 + loss2 + loss3 + loss4

        if not return_dict:
            output = (logits1, logits2, logits3, logits4, ) + encoder_outputs[2:]
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
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class T5EncoderArcFaceForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.classifier1 = ArcFace(config.d_model, config.category1_num_labels)
        self.classifier2 = ArcFace(config.d_model, config.category2_num_labels)
        self.classifier3 = ArcFace(config.d_model, config.category3_num_labels)
        self.classifier4 = ArcFace(config.d_model, config.category4_num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels1:Optional[torch.FloatTensor] = None,
        labels2:Optional[torch.FloatTensor] = None,
        labels3:Optional[torch.FloatTensor] = None,
        labels4:Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        sequence_output = hidden_states[input_ids == self.config.eos_token_id]
        sequence_output = self.dropout(sequence_output)

        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)
        logits3 = self.classifier3(sequence_output)
        logits4 = self.classifier4(sequence_output)

        loss = None
        loss1, loss2, loss3, loss4 = None, None, None, None
        if labels1 is not None and labels2 is not None and labels3 is not None and labels4 is not None:
            
            arc_logits1 = self.classifier1(sequence_output, labels1)
            arc_logits2 = self.classifier2(sequence_output, labels2)
            arc_logits3 = self.classifier3(sequence_output, labels3)
            arc_logits4 = self.classifier4(sequence_output, labels4)

            loss_fct = nn.CrossEntropyLoss()
            loss1 = loss_fct(arc_logits1.view(-1, self.config.category1_num_labels), labels1.view(-1))
            loss2 = loss_fct(arc_logits2.view(-1, self.config.category2_num_labels), labels2.view(-1))
            loss3 = loss_fct(arc_logits3.view(-1, self.config.category3_num_labels), labels3.view(-1))
            loss4 = loss_fct(arc_logits4.view(-1, self.config.category4_num_labels), labels4.view(-1))

            loss = (loss1 + loss2 + loss3 + loss4) / 4

        if not return_dict:
            output = (logits1, logits2, logits3, logits4, ) + encoder_outputs[2:]
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
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
