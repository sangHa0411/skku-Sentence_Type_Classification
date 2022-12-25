# skku-imbalance_classification

## Project
  * 대회 : 2022 성균관대 주최 문장 유형 분류 AI 경진대회
  * 목표
      1. 불균형 극심한 데이터셋 과정에서 어떻게 하면 Classification 성능을 높일 수 있는지 알아보기
      
## Version
  * Torch version : 1.11.0
  * Numpy version : 1.23.4
  * Pandas version : 1.5.2
  * Datasets version : 2.3.0
  * Transformers version : 4.19.0
  
## Platform
  * Google Cloud Platform - A100 GPU (Ubuntu)
      
## Dataset
  * 문장 유형에 관한 분류 데이터
      1. 4가지 카테고리로 분류되어 있다.
          1. 유형, 극성, 시제, 확실성
      2. 4가지 카테고리를 각각 분류해서 전체 하나의 카테고리로 분류해야한다.
          1. 분류 유형 : 64가지
  * 데이터셋 불균형 정도
      * 가장 많은 데이터 수를 가지고 있는 Label을 4000개 ~ 7000개가 있으며 데이터 수가 1개인 Label도 상당 수 존재하였다.
      * ![스크린샷 2022-12-25 오후 1 07 13](https://user-images.githubusercontent.com/48673702/209456675-8899037c-90ba-45c8-9760-c01058ae37b5.png)


## Baseline
  1. PLM : klue/roberta-large
  2. Model Structure
      * CLS 토큰 이외에도 EOS 토큰의 출력을 Concat해서 Sequence Output으로 설정
  3. Training : R-Drop Loss
 
## Meaningful Method
  1. Data Augmentation
      1. 데이터 수가 20개가 안되는 Label들을 대상으로 다양한 방법을 통해서 데이터를 증강 최소 20개로 만들어주었다.
          1. 문장 내에서 임의 단어 삭제
          2. 문장 내에서 임의 단어 변환
          3. 문장의 순서 변환
          4. 문장 길이에 따른 문장에 구두점 삽입 (aeda)
  2. Undersampling
      1. 데이터 수가 3000개 이상인 Label에 대해서는 해당 데이터의 80%만을 사용해서 모델을 학습 
  3. Stratified Sampling
      1. Label에 따른 데이터 수의 비율에 맞게 학습 Batch안에 들어가는 데이터를 지정
      2. 학습 Batch안에 모든 데이터의 Label이 똑같거나 대부분인 경우를 방지

## Hyperparameter
|Hyperparameter|Value|
|--------|-----------|
|PLM|klue/roberta-large|
|epochs|3|
|learning rate|2.5e-5|
|max length|128|
|train batch size|16|
|weight decay|1e-3|
|warmup ratio|0.05|
|miminum_size|20|
|maximum_size|20|


## Terminal Command Example
  ```shell
  # Training & Evaluation
  python train.py \
  --do_eval \
  --seed 42 \
  --PLM klue/roberta-large \
  --model_name RobertaSpecialTokenForSequenceClassification \
  --validation_ratio 0.2 \
  --max_length 128 \
  --output_dir exps/test \
  --data_dir data \
  --train_data_file train.csv \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --eval_steps 500 \
  --logging_steps 100 \
  --evaluation_strategy steps \
  --save_strategy no \
  --learning_rate 2.5e-5 \
  --warmup_ratio 0.05 \
  --minimum_size 20 \
  --maximum_size 3000
  
  # Training
  python train.py \
  --PLM klue/roberta-large \
  --seed 1 \
  --max_length 128 \
  --output_dir exps/model1 \
  --data_dir data \
  --model_name RobertaSpecialTokenForSequenceClassification \
  --train_data_file train.csv \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --save_steps 500 \
  --logging_steps 100 \
  --evaluation_strategy no \
  --save_strategy steps \
  --learning_rate 2.5e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-3 \
  --minimum_size 20 \
  --maximum_size 3000
  
  python train.py \
  --PLM klue/roberta-large \
  --seed 2 \
  --max_length 128 \
  --output_dir exps/model2 \
  --data_dir data \
  --model_name RobertaSpecialTokenForSequenceClassification \
  --train_data_file train.csv \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --save_steps 500 \
  --logging_steps 100 \
  --evaluation_strategy no \
  --save_strategy steps \
  --learning_rate 2.5e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-3 \
  --minimum_size 20 \
  --maximum_size 3000
  
  python train.py \
  --PLM klue/roberta-large \
  --seed 3 \
  --max_length 128 \
  --output_dir exps/model3 \
  --data_dir data \
  --model_name RobertaSpecialTokenForSequenceClassification \
  --train_data_file train.csv \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --save_steps 500 \
  --logging_steps 100 \
  --evaluation_strategy no \
  --save_strategy steps \
  --learning_rate 2.5e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-3 \
  --minimum_size 20 \
  --maximum_size 3000
  
  python train.py \
  --PLM klue/roberta-large \
  --seed 4 \
  --max_length 128 \
  --output_dir exps/model4 \
  --data_dir data \
  --model_name RobertaSpecialTokenForSequenceClassification \
  --train_data_file train.csv \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --save_steps 500 \
  --logging_steps 100 \
  --evaluation_strategy no \
  --save_strategy steps \
  --learning_rate 2.5e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-3 \
  --minimum_size 20 \
  --maximum_size 3000
  
  python train.py \
  --PLM klue/roberta-large \
  --seed 5 \
  --max_length 128 \
  --output_dir exps/model5 \
  --data_dir data \
  --model_name RobertaSpecialTokenForSequenceClassification \
  --train_data_file train.csv \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --save_steps 500 \
  --logging_steps 100 \
  --evaluation_strategy no \
  --save_strategy steps \
  --learning_rate 2.5e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-3 \
  --minimum_size 20 \
  --maximum_size 3000
  
  # Ensemble
  ## Training Model with different seeds (1,2,3,4,5), Other hyperparameters are same
  python ensemble.py \
  --data_dir data \
  --train_data_file train.csv \
  --test_data_file test.csv \
  --max_length 128 \
  --per_device_eval_batch_size 16 \
  --submission_data_file sample_submission.csv \
  --save_dir results \
  --save_file_name restoration.csv
 
  ```

## Validation
  * 각 Label별로 데이터 수의 20%을 분리해서 Train, Validation 데이터를 만들고 모델 성능 검증
  * Seed 42
  * 성능 평가
      * ![스크린샷 2022-12-25 오후 1 35 53](https://user-images.githubusercontent.com/48673702/209457065-320e552f-8a93-4a46-a3e7-4efbc329d524.png)  


## Ensemble
  * 위 hyperparameter의 모델을 seed를 다르게 해서 5개의 모델을 학습하고 Soft Voting을 실행

## Leaderboard
|Model|Public|Private|
|-----|----|----|
|ensemable|0.75731(7th)|0.75506(6th)|

## 후기
  * 배운 점 
      1. Back Translation 방법과 같이 비용이 많이 드는 데이터 증강 방법 이외에도 다양한 방법으로 데이터 증강을 할 수가 있고 이 방법들이 실제로 효력이 있었다.
      2. Undersampling을 적용하였을 때 모델 성능이 더 좋아졌고 데이터를 임의로 선택하는 과정 덕분에 앙상블에서 많은 성능 향상 있었다.
      3. 데이터의 비율에 맞게 sampler를 따로 만들어서 학습에 적용하는 것도 모델 성능 향상에 도움이 되었다.
      


