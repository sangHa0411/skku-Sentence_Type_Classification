
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, List, Any, Union, Tuple, Optional
from transformers.trainer_pt_utils import nested_detach
from utils.sampler import StratifiedSampler

class Trainer(Trainer) :

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:

        generator = torch.Generator()
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        train_batch_size = self.args.per_device_train_batch_size
        return StratifiedSampler(self.train_dataset, batch_size=train_batch_size, generator=generator)
        