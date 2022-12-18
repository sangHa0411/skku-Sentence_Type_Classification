import random
import numpy as np
import pandas as pd
import collections
from datasets import Dataset
from tqdm import tqdm

class Augmentation :

    def __init__(self, min_num, reduction=0.8, eval_flag=True) :
        self.min_num = min_num
        self.reduction = reduction
        self.punct = [".", ";", "?", ":", "!", ","]
        self.eval_flag = eval_flag

    def __call__(self, dataset):
        
        labels = dataset['label']

        label_ids = collections.defaultdict(list)
        for i, l in enumerate(labels) :
            label_ids[l].append(i)

        total_id_list = []
        for l in label_ids :
            id_list = label_ids[l]
            previous_size = len(id_list)

            if previous_size > self.min_num :                
                if self.eval_flag :
                    augmentated_id_list = id_list
                else :
                    sample_size = int(len(id_list) * self.reduction)
                    augmentated_id_list = random.sample(id_list, sample_size)
                    
                augmentated_id_list = [(a_id, 0) for a_id in augmentated_id_list]
            else :
                augmentated_id_list = [(a_id, 0) for a_id in id_list]
                while len(id_list) < self.min_num :
                    id_list = id_list * 2

                augmentated_id_list = augmentated_id_list + \
                    [(a_id, 1) for a_id in random.sample(id_list, self.min_num - previous_size)]

            print('Category : %s \t Previous size : %d, Current size : %d' %(l, previous_size, len(augmentated_id_list)))

            total_id_list.extend(augmentated_id_list)

        org_size, add_size = 0, 0
        for _, status in total_id_list :
            if status == 0 :
                org_size += 1
            else :
                add_size += 1
        print('Original data size : %d \t Augment data size : %d' %(org_size, add_size))

        random.shuffle(total_id_list)
            
        total_dataset = []
        for i, status in tqdm(total_id_list) :
            data = self.augment(dataset[i], status)
            total_dataset.append(data)

        df = pd.DataFrame(total_dataset)
        dataset = Dataset.from_pandas(df)
        return dataset


    def aeda(self, data) :
        sentence = data['문장']
        insert_size = np.random.randint(1, len(sentence) // 3)
        
        chars = list(sentence)
        while insert_size > 0 :
            punct_id = np.random.randint(len(self.punct))
            punct = self.punct[punct_id]

            insert_id = np.random.randint(len(chars))
            chars = chars[:insert_id] + [punct] + chars[insert_id:]

            insert_size -= 1

        sentence = ''.join(chars)
        data['문장'] = sentence
        return data


    def reverse(self, data) :
        sentence = data['문장']
        words = sentence.split(' ')

        if len(words) > 5 :   
            index = np.random.randint(1, len(words) - 1)
            reversed = words[index:] + words[:index]
            sentence = ' '.join(reversed)

        data['문장'] = sentence
        return data


    def delete(self, data) :

        sentence = data['문장']
        words = sentence.split(' ')

        if len(words) > 5 :   
            word_size = len(words)
            del_size = int(word_size * 0.2)
            del_indices = random.sample(range(word_size), del_size)

            deleted = []
            for i, word in enumerate(words) :
                if i in del_indices :
                    continue
                deleted.append(word)

            sentence = ' '.join(deleted)

        data['문장'] = sentence
        return data


    def augment(self, data, status) :
        if status == 0 :
            return data
        else :
            option = np.random.randint(3)
            if option == 0 :
                data = self.aeda(data)
            elif option == 1 :
                data = self.reverse(data)
            else :
                data = self.delete(data)
            return data

