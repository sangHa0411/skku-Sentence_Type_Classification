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

        # 각 label별로 데이터 인덱스들을 정리
        total_size = len(self.data_source['labels'])
        labels = self.data_source['labels']
        label_indices = self.get_label_indices(labels)

        # 각 label이 가지는 데이터 수를 파악
        label_batch_sizes = {}
        for l in label_indices :
            l_indices = label_indices[l]
            label_batch_sizes[l] = len(l_indices)
        
        # 데이터 수를 많이 가지고 있는 label들을 파악
        label_items = sorted(label_batch_sizes.items(), key=lambda x : x[1], reverse=True)
        iter_size = total_size // self.batch_size

        # Label 1 : 사실형-긍정-미래-확실
        item1 = label_items[0][0]
        item1_size = len(label_indices[item1])
        item1_batch_size = item1_size // iter_size
        item1_indices = label_indices[item1]

        # Label 2 : 사실형-긍정-과거-확실
        item2 = label_items[1][0]
        item2_size = len(label_indices[item2])
        item2_batch_size = item2_size // iter_size
        item2_indices = label_indices[item2]

        # Label 1, 2 제외한 나머지 label들
        item3_size = total_size - (item1_size + item2_size)
        item3_batch_size = item3_size // iter_size
        item3_indices = []
        for l in label_indices :
            if l != item1 and l != item2 :
                item3_indices.extend(label_indices[l])
        random.shuffle(item3_indices)

        # 3개의 그룹으로 나뉘어진 인덱스들을 각 비율에 맞게 batch에 들어가게끔 한다.
        """
        예시
        train batch size : 16

            1) Label 1 (사실형-긍정-미래-확실) : 6
            2) Label 2 (사실형-긍정-과거-확실) : 5
            3) 그 외 나머지 : 5
        
        하나의 batch안에 모두 같은 label이 되는 것을 방지한다.
        """
        sampled = []
        for i in range(iter_size) :
            sub_sampled = []

            indices = item1_indices[i*item1_batch_size:(i+1)*item1_batch_size]
            if len(indices) > 0 :
                sub_sampled.extend(indices)

            indices = item2_indices[i*item2_batch_size:(i+1)*item2_batch_size]
            if len(indices) > 0 :
                sub_sampled.extend(indices)
    
            indices = item3_indices[i*item3_batch_size:(i+1)*item3_batch_size]
            if len(indices) > 0 :
                sub_sampled.extend(indices)

            random.shuffle(sub_sampled)
            sampled.extend(sub_sampled)

        remain_ids = []

        remain_ids.extend(item1_indices[item1_batch_size*iter_size:])
        remain_ids.extend(item2_indices[item2_batch_size*iter_size:])
        remain_ids.extend(item3_indices[item3_batch_size*iter_size:])

        if len(remain_ids) > 0 :
            random.shuffle(remain_ids)
            sampled.extend(remain_ids)

        yield from sampled

    def __len__(self) -> int:
        return self.num_samples