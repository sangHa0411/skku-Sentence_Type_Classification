import random
import numpy as np
import pandas as pd
import collections
from datasets import Dataset
from tqdm import tqdm

class Augmentation :

    def __init__(self, tokenizer, max_num, min_num, reduction=0.8, undersampling_ratio=0.8, eval_flag=True) :
        self.tokenizer = tokenizer
        self.max_num = max_num
        self.min_num = min_num
        self.reduction = reduction
        self.undersampling_ratio = undersampling_ratio
        self.punct = [".", ";", "?", ":", "!", ","]
        self.eval_flag = eval_flag

    def __call__(self, dataset):
        
        labels = dataset['label']

        # 각 Label별로 Index들을 정리한다.
        label_ids = collections.defaultdict(list)
        for i, l in enumerate(labels) :
            label_ids[l].append(i)

        # 각 Label 별로 데이터 크기에 따라서 undersampling, dataaugmentation을 따로 진행한다.
        total_id_list = []
        for l in label_ids :
            id_list = label_ids[l]
            previous_size = len(id_list) # 기존 데이터 갯수

            # 최소 갯수 보다 많으면 데이터 증강을 하지 않음 (20개)
            if previous_size > self.min_num : 
                if self.eval_flag :
                    # validation을 할 때는 이미 Validation을 할 때 20%를 각 Label에서 가져가끼 때문에 그대로
                    augmentated_id_list = id_list
                else :
                    # traina만 할 때는 validation과 같은 환경을 맞추기 위해서 각 전체 데이터 수의 80%만을 선정
                    ## 전체 데이터를 다 쓰는 것보다 특정 비율을 가지고 임의로 선택하고 모델들을 앙상블 했을 때 더 좋은 효과를 가질 것이라고 판단
                    sample_size = int(len(id_list) * self.reduction)
                    augmentated_id_list = random.sample(id_list, sample_size)

                # 최대 갯수보다 많으면 (3000개)
                if len(augmentated_id_list) > self.max_num : 
                    # 데이터 수의 80%를 임의 선택
                    augmentated_id_list = random.sample(
                        augmentated_id_list, 
                        int(len(augmentated_id_list) * self.undersampling_ratio)
                    )

                # (index, status) 구조로 저장
                # status 0 : 데이터 변형 대상 아님, status : 1 데이터 변형 대상
                augmentated_id_list = [(a_id, 0) for a_id in augmentated_id_list]
            
            # 최소 갯수 보다 많으면 데이터 증강을 진행 (20개)
            else :
                # 기존 데이터
                augmentated_id_list = [(a_id, 0) for a_id in id_list]
                # 기존 데이터를 복사 (20개가 넘을 떼 까지)
                while len(id_list) < self.min_num :
                    id_list = id_list * 2

                # 기존 데이터 + 복사된 변형 될 데이터 해서 20개를 맞춘다.
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

        # 데이터 셔플
        random.shuffle(total_id_list)
            
        # huggingface를 사용하기 때문에 datasets class 형식으로 맞춤
        total_dataset = []
        for i, status in tqdm(total_id_list) :
            data = self.augment(dataset[i], status)
            total_dataset.append(data)

        df = pd.DataFrame(total_dataset)
        dataset = Dataset.from_pandas(df)
        return dataset

    # aeda를 통한 문장 변형
    def aeda(self, data) :
        sentence = data['문장']
        # 구두점 삽입 횟수
        insert_size = np.random.randint(1, len(sentence) // 3)
        
        chars = list(sentence)
        while insert_size > 0 :
            # 구두점 선택
            punct_id = np.random.randint(len(self.punct))
            punct = self.punct[punct_id]

            # 구두점 삽입 위치 선정
            insert_id = np.random.randint(len(chars))
            # 구두점 삽입
            chars = chars[:insert_id] + [punct] + chars[insert_id:]

            insert_size -= 1

        sentence = ''.join(chars)
        data['문장'] = sentence
        return data

    # 문장 내의 몇몇 단어들을 임의의 단어로 변환
    def change(self, data) :
        sentence = data['문장']
        # tokenizer를 통해서 index들로 변환
        tokens = self.tokenizer.encode(sentence)[1:-1]

        # 바꾸는 단어 수 선택 (문장의 길이에 따라서 달라진다.)
        change_size = int(len(tokens) * 0.15)
        
        if change_size > 0 :
            change_ids = random.sample(range(len(tokens)), change_size)
            for c_id in change_ids :
                tokens[c_id] = np.random.randint(len(self.tokenizer))

            data['문장'] = self.tokenizer.decode(tokens)

        return data

    # 문장의 순서 바꾸기
    """
    예시
        before : '그렇다면 차천로는 과연 어떤 비석을 두고 이런 말을 했던 걸까.'
        after : '비석을 두고 이런 말을 했던 걸까. 그렇다면 차천로는 과연 어떤'
    """
    def reverse(self, data) :
        sentence = data['문장']
        words = sentence.split(' ')

        if len(words) > 5 :   
            index = np.random.randint(1, len(words) - 1)
            reversed = words[index:] + words[:index]
            sentence = ' '.join(reversed)

        data['문장'] = sentence
        return data

    # 문장의 단어 삭제
    """
    예시
        before : '이것이 만약 상업적인 의도였다면 이렇게까지 하지 못했을 것이다.'
        after : '이것이 만약 상업적인 의도였다면 이렇게까지 하지 것이다.'
    """
    def delete(self, data) :

        sentence = data['문장']
        words = sentence.split(' ')

        if len(words) > 5 :   
            word_size = len(words)
            # 삭제 단어 횟수 선정 (문장의 길이에 따라서 정해진다.)
            del_size = int(word_size * 0.2)
            # 삭제 단어 위치 선정
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
            option = np.random.randint(4)
            if option == 0 :
                data = self.aeda(data)
            elif option == 1 :
                data = self.reverse(data)
            elif option == 2 :
                data = self.delete(data)
            else :
                data = self.change(data)
            
            return data

