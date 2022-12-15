import math
import torch
import random
import collections
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

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        label_map = collections.defaultdict(list)
        labels1 = self.data_source['labels1']
        labels2 = self.data_source['labels2']
        labels3 = self.data_source['labels3']
        labels4 = self.data_source['labels4']
        
        total_size = len(labels1)
        for i in range(total_size) :
            label_list = [labels1[i], labels2[i], labels3[i], labels4[i]]
            label_str = '-'.join(map(str, label_list))

            label_map[label_str].append(i)
        
        for l in label_map :
            random.shuffle(label_map[l])

        label_size = {}
        for l in label_map :
            label_size[l] = int(self.batch_size * (len(label_map[l]) / total_size))
            if label_size[l] == 0 :
                label_size[l] = 1

        sampled = []
        iter_size = n // self.batch_size
        for i in range(iter_size) :
            sub_sampled = []
            for l in label_map :
                sub_batch_size = label_size[l]

                if sub_batch_size*i > len(label_map[l]) :
                    continue

                sub_ids = label_map[l][sub_batch_size*i:sub_batch_size*(i+1)]
                sub_sampled.extend(sub_ids)

            random.shuffle(sub_sampled)
            sampled.extend(sub_sampled)

        remain_ids = []
        for l in label_map :
            remain_ids.extend(label_map[l][iter_size*label_size[l]:])
        random.shuffle(remain_ids)
        sampled.extend(remain_ids)

        yield from sampled

    def __len__(self) -> int:
        return self.num_samples