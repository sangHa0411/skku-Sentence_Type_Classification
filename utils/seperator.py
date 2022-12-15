import random
from collections import defaultdict
from datasets import Dataset, DatasetDict

class Seperator :

    def __init__(self, validation_ratio) :
        self.validation_ratio = validation_ratio

    def __call__(self, df) :
        org_dataset = Dataset.from_pandas(df)

        mapping = defaultdict(list)
        labels = list(df['label'])

        for i, l in enumerate(labels) :
            mapping[l].append(i)
        
        for l in mapping :
            random.shuffle(mapping[l])

        train_ids, validation_ids = [], []

        for l in mapping :
            id_list = mapping[l]

            if len(id_list) >= 5 :
                size = len(id_list)
                train_size = int(size * 0.8)
                train_ids.extend(id_list[:train_size])
                validation_ids.extend(id_list[train_size:])
            else :
                validation_ids.extend(id_list)

        train_dset = org_dataset.select(train_ids)
        validation_dset = org_dataset.select(validation_ids)

        dset = DatasetDict({'train' : train_dset, 'validation' : validation_dset})
        return dset
