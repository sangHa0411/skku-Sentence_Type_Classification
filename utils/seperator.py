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

        # 각 레이블마다 데이터 인덱스들을 따로 저장하면서 정리
        for i, l in enumerate(labels) :
            mapping[l].append(i)
        
        for l in mapping :
            random.shuffle(mapping[l])

        # 각 Label 별로 총 데이터의 20%를 분리하면서 Validation data를 만든다.
        train_ids, validation_ids = [], []

        for l in mapping :
            id_list = mapping[l]

            if len(id_list) >= 5 :
                size = len(id_list)
                train_size = int(size * (1.0-self.validation_ratio))
                train_ids.extend(id_list[:train_size])
                validation_ids.extend(id_list[train_size:])
            else :
                train_ids.extend(id_list)

        random.shuffle(train_ids)
        random.shuffle(validation_ids)

        train_dset = org_dataset.select(train_ids)
        validation_dset = org_dataset.select(validation_ids)

        # Datasets 포멧으로 변경
        dset = DatasetDict({'train' : train_dset, 'validation' : validation_dset})
        return dset
