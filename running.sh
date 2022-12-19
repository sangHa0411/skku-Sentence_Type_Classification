# Training & Evaluation
## Roberta Model

### Submission 1
python train.py \
--do_eval \
--seed 42 \
--PLM klue/roberta-large \
--model_name RobertaBaseForSequenceClassification \
--validation_ratio 0.2 \
--max_length 128 \
--output_dir exps/test \
--data_dir data \
--train_data_file train.csv \
--num_train_epochs 3 \
--max_steps 2000 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--eval_steps 500 \
--logging_steps 100 \
--evaluation_strategy steps \
--save_strategy no \
--learning_rate 5e-6 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 20

### Submission 2
python train.py \
--do_eval \
--seed 42 \
--PLM klue/roberta-large \
--model_name RobertaBaseForSequenceClassification \
--validation_ratio 0.2 \
--max_length 128 \
--output_dir exps/test \
--data_dir data \
--train_data_file train.csv \
--num_train_epochs 3 \
--max_steps 2000 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--eval_steps 500 \
--logging_steps 100 \
--evaluation_strategy steps \
--save_strategy no \
--learning_rate 1e-5 \
--warmup_ratio 0.05 \
--weight_decay 1e-4 \
--minimum_size 20

### Submission 3
python train.py \
--do_eval \
--seed 42 \
--PLM klue/roberta-large \
--model_name RobertaBaseForSequenceClassification \
--validation_ratio 0.2 \
--max_length 128 \
--output_dir exps/test \
--data_dir data \
--train_data_file train.csv \
--num_train_epochs 3 \
--max_steps 2500 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--eval_steps 500 \
--logging_steps 100 \
--evaluation_strategy steps \
--save_strategy no \
--learning_rate 5e-6 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 30

## Submission 4
python train.py \
--PLM KETI-AIR/ke-t5-large-ko \
--seed 42 \
--max_length 128 \
--validation_ratio 0.2 \
--do_eval \
--output_dir exps/model3 \
--data_dir data \
--model_name T5EncoderFocalForSequenceClassification \
--train_data_file train.csv \
--num_train_epochs 3 \
--max_steps 2000 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--eval_steps 500 \
--logging_steps 100 \
--evaluation_strategy steps \
--save_strategy no \
--learning_rate 5e-5 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 20

# # Full Training
# ## Model1
# python train.py \
# --PLM klue/roberta-large \
# --seed 42 \
# --max_length 128 \
# --output_dir exps/model1 \
# --data_dir data \
# --model_name RobertaBaseForSequenceClassification \
# --train_data_file train.csv \
# --num_train_epochs 3 \
# --max_steps 2000 \
# --per_device_train_batch_size 16 \
# --save_steps 500 \
# --logging_steps 100 \
# --evaluation_strategy no \
# --save_strategy steps \
# --learning_rate 1e-5 \
# --warmup_ratio 0.05 \
# --weight_decay 1e-4 \
# --minimum_size 20

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
# --model_path exps/model1/checkpoint-1500 \
# --max_length 128 \
# --per_device_eval_batch_size 16 \
# --submission_data_file sample_submission.csv \
# --save_dir results \
# --save_file_name 2022-12-19-02.csv

# # Ensemble
# python ensemble.py \
# --data_dir data \
# --test_data_file test.csv \
# --max_length 128 \
# --per_device_eval_batch_size 16 \
# --submission_data_file sample_submission.csv \
# --save_dir results \
# --save_file_name 2022-12-18-03.csv