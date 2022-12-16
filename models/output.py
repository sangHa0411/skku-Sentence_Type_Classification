import torch
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple

@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    category1_loss: Optional[torch.FloatTensor] = None
    category2_loss: Optional[torch.FloatTensor] = None
    category3_loss: Optional[torch.FloatTensor] = None
    category4_loss: Optional[torch.FloatTensor] = None
    category1_logits: torch.FloatTensor = None
    category2_logits: torch.FloatTensor = None
    category3_logits: torch.FloatTensor = None
    category4_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    