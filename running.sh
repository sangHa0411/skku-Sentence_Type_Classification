# Training & Evaluation
## Roberta Model
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
--learning_rate 3e-5 \
--weight_decay 1e-3 \
--warmup_ratio 0.05 \
--minimum_size 20 \
--maximum_size 3000

## Electra
python train.py \
--do_eval \
--seed 42 \
--PLM tunib/electra-ko-base \
--model_name ElectraForSequenceClassification \
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
--learning_rate 3e-5 \
--weight_decay 1e-3 \
--warmup_ratio 0.05 \
--minimum_size 20 \
--maximum_size 3000

# Full Training
## Model1
python train.py \
--PLM klue/roberta-large \
--seed 1 \
--max_length 128 \
--output_dir exps/model1 \
--data_dir data \
--model_name RobertaBaseForSequenceClassification \
--train_data_file train.csv \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--save_steps 500 \
--logging_steps 100 \
--evaluation_strategy no \
--save_strategy steps \
--learning_rate 7e-6 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 30 \
--maximum_size 3000


python train.py \
--PLM klue/roberta-large \
--seed 2 \
--max_length 128 \
--output_dir exps/model2 \
--data_dir data \
--model_name RobertaBaseForSequenceClassification \
--train_data_file train.csv \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--save_steps 500 \
--logging_steps 100 \
--evaluation_strategy no \
--save_strategy steps \
--learning_rate 7e-6 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 30 \
--maximum_size 3000


python train.py \
--PLM klue/roberta-large \
--seed 3 \
--max_length 128 \
--output_dir exps/model3 \
--data_dir data \
--model_name RobertaBaseForSequenceClassification \
--train_data_file train.csv \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--save_steps 500 \
--logging_steps 100 \
--evaluation_strategy no \
--save_strategy steps \
--learning_rate 7e-6 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 30 \
--maximum_size 3000


python train.py \
--PLM klue/roberta-large \
--seed 4 \
--max_length 128 \
--output_dir exps/model4 \
--data_dir data \
--model_name RobertaBaseForSequenceClassification \
--train_data_file train.csv \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--save_steps 500 \
--logging_steps 100 \
--evaluation_strategy no \
--save_strategy steps \
--learning_rate 7e-6 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 30 \
--maximum_size 3000


python train.py \
--PLM klue/roberta-large \
--seed 5 \
--max_length 128 \
--output_dir exps/model5 \
--data_dir data \
--model_name RobertaBaseForSequenceClassification \
--train_data_file train.csv \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--save_steps 500 \
--logging_steps 100 \
--evaluation_strategy no \
--save_strategy steps \
--learning_rate 7e-6 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 30 \
--maximum_size 3000

# ## Model2
# python train.py \
# --PLM klue/roberta-large \
# --seed 2 \
# --max_length 128 \
# --output_dir exps/model2 \
# --data_dir data \
# --model_name RobertaFocalForSequenceClassification \
# --train_data_file train.csv \
# --num_train_epochs 3 \
# --max_steps 2000 \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 16 \
# --save_steps 500 \
# --logging_steps 100 \
# --evaluation_strategy no \
# --save_strategy steps \
# --learning_rate 5e-6 \
# --warmup_ratio 0.05 \
# --weight_decay 1e-3 \
# --minimum_size 20 \
# --maximum_size 3000

# ## Model3
# python train.py \
# --PLM KETI-AIR/ke-t5-large-ko \
# --seed 3 \
# --max_length 128 \
# --output_dir exps/model3 \
# --data_dir data \
# --model_name T5EncoderArcFaceForSequenceClassification \
# --train_data_file train.csv \
# --num_train_epochs 3 \
# --max_steps 2000 \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 16 \
# --save_steps 500 \
# --logging_steps 100 \
# --evaluation_strategy no \
# --save_strategy steps \
# --learning_rate 5e-5 \
# --warmup_ratio 0.05 \
# --weight_decay 1e-3 \
# --minimum_size 20 \
# --maximum_size 3000

# # Predict
# python predict.py \
# --data_dir data \
# --test_data_file test.csv \
# --model_type roberta \
# --model_name RobertaBaseForSequenceClassification \
# --model_path exps/model1/ \
# --max_length 128 \
# --per_device_eval_batch_size 16 \
# --submission_data_file sample_submission.csv \
# --save_dir results \
# --save_file_name 2022-12-20-01.csv

# Ensemble
python ensemble.py \
--data_dir data \
--test_data_file test.csv \
--max_length 128 \
--per_device_eval_batch_size 16 \
--submission_data_file sample_submission.csv \
--save_dir results \
--save_file_name 2022-12-21-02.csv