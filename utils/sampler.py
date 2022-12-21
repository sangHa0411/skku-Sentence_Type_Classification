import math
import torch
import random
import collections
import numpy as np
from torch.utils.data.sampler import Sampler
from typing import Iterator, Optional, Sized


class StratifiedSampler(Sampler[int]):

    data_source: Sized
    replacement: bool

    def __init__(
        self, 
        data_source: Sized, 
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = None, 
        generator=None
    ) -> None:

        self.data_source = data_source
        self.batch_size = batch_size
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size should be a positive integer "
                             "value, but got batch_size={}".format(self.batch_size))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def get_label_indices(self, labels) :
        label_indices = collections.defaultdict(list)
        
        total_size = len(labels)
        for i in range(total_size) :
            label = labels[i]
            label_indices[label].append(i)
        
        for l in label_indices :
            random.shuffle(label_indices[l])

        return label_indices 

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        total_size = len(self.data_source['labels'])
        labels = self.data_source['labels']
        label_indices = self.get_label_indices(labels)

        label_batch_sizes = {}
        for l in label_indices :
            l_indices = label_indices[l]
            label_batch_sizes[l] = (len(l_indices) / total_size) * self.batch_size

        breakpoint()

        for l in label_batch_size :
            if label_batch_size[l] > 1 :
                major_label_indices.extend(label_indices[l])
            else :
                minor_label_indices.extend(label_indices[l])

        random.shuffle(major_label_indices)
        random.shuffle(minor_label_indices)

        sampled = []


        yield from sampled

    def __len__(self) -> int:
        return self.num_samples