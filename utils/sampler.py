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

    def get_label_indices(self, data_socre) :

        label_indices = collections.defaultdict(list)
        labels1 = data_socre['labels1']
        labels2 = data_socre['labels2']
        labels3 = data_socre['labels3']
        labels4 = data_socre['labels4']
        
        total_size = len(labels1)
        for i in range(total_size) :
            label_list = [labels1[i], labels2[i], labels3[i], labels4[i]]
            label_str = '-'.join(map(str, label_list))
            label_indices[label_str].append(i)
        
        for l in label_indices :
            random.shuffle(label_indices[l])

        label_batch_size = {}
        for l in label_indices :
            sub_label_size = int(self.batch_size * (len(label_indices[l]) / total_size))
            label_batch_size[l] = sub_label_size

        return label_indices, label_batch_size 

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        total_size = len(self.data_source['labels1'])
        label_indices, label_batch_size = self.get_label_indices(self.data_source)

        major_label_indices = []
        minor_label_indices = []

        for l in label_batch_size :
            if label_batch_size[l] > 1 :
                major_label_indices.extend(label_indices[l])
            else :
                minor_label_indices.extend(label_indices[l])

        random.shuffle(major_label_indices)
        random.shuffle(minor_label_indices)

        sampled = []
        iter_size = total_size // self.batch_size
        major_batch_size = len(major_label_indices) // iter_size
        minor_batch_size = len(minor_label_indices) // iter_size 
        
        for i in range(iter_size) :
            sub_sampled = []

            if major_batch_size*i < len(major_label_indices) :
                sub_sampled.extend(major_label_indices[major_batch_size*i:major_batch_size*(i+1)])

            if minor_batch_size*i < len(minor_label_indices) :
                sub_sampled.extend(minor_label_indices[minor_batch_size*i:minor_batch_size*(i+1)])

            random.shuffle(sub_sampled)
            sampled.extend(sub_sampled)

        remain_ids = []
        if major_batch_size * iter_size < len(major_label_indices) :
            remain_ids.extend(major_label_indices[major_batch_size*iter_size:])

        if minor_batch_size * iter_size < len(minor_label_indices) :
            remain_ids.extend(minor_label_indices[minor_batch_size*iter_size:])
        
        if len(remain_ids) > 0 :
            random.shuffle(remain_ids)
            sampled.extend(remain_ids)

        yield from sampled

    def __len__(self) -> int:
        return self.num_samples