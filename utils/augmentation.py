import random
import numpy as np
import pandas as pd
import collections
from datasets import Dataset
from tqdm import tqdm

class Augmentation :

    def __init__(self, max_num, min_num) :
        self.max_num = max_num
        self.min_num = min_num

    def __call__(self, dataset):
        
        labels = dataset['label']

        label_ids = collections.defaultdict(list)
        for i, l in enumerate(labels) :
            label_ids[l].append(i)

        total_id_list = []
        for l in label_ids :
            id_list = label_ids[l]
            previous_size = len(id_list)

            if len(id_list) > self.min_num :
                if len(id_list) > self.max_num :
                    augmentated_id_list = random.sample(id_list, self.max_num)
                else :
                    augmentated_id_list = id_list
            else :
                while len(id_list) < self.min_num :
                    id_list = id_list * 2
                augmentated_id_list = random.sample(id_list, self.min_num)
            print('Category : %s \t Previous size : %d, Current size : %d' %(l, previous_size, len(augmentated_id_list)))

            total_id_list.extend(augmentated_id_list)

        random.shuffle(total_id_list)
            
        total_dataset = []
        for i in tqdm(total_id_list) :
            total_dataset.append(dataset[i])

        df = pd.DataFrame(total_dataset)
        dataset = Dataset.from_pandas(df)
        return dataset
