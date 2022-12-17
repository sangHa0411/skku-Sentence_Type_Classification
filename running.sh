# Training & Evaluation
## Roberta Model
python train.py \
--do_eval \
--seed 42 \
--PLM klue/roberta-large \
--model_name RobertaFocalForSequenceClassification \
--validation_ratio 0.2 \
--max_length 128 \
--output_dir exps \
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
--minimum_size 20 \
--maximum_size 5000

## Electra Model
python train.py \
--do_eval \
--seed 42 \
--PLM tunib/electra-ko-base \
--model_name ElectraForSequenceClassification \
--validation_ratio 0.2 \
--max_length 128 \
--output_dir exps \
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
--learning_rate 1e-5 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 30 \
--maximum_size 5000

## T5 Encoder
python train.py \
--do_eval \
--seed 42 \
--PLM KETI-AIR/ke-t5-large-ko \
--model_name T5EncoderForSequenceClassification \
--validation_ratio 0.2 \
--max_length 128 \
--output_dir exps \
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
--minimum_size 20 \
--maximum_size 5000

# Full Training
python train.py \
--PLM klue/roberta-large \
--seed 42 \
--max_length 128 \
--output_dir exps \
--data_dir data \
--model_name RobertaBaseForSequenceClassification \
--train_data_file train.csv \
--num_train_epochs 3 \
--max_steps 2000 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--save_steps 500 \
--logging_steps 100 \
--evaluation_strategy no \
--save_strategy steps \
--learning_rate 5e-6 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 20 \
--maximum_size 5000

# Predict
python predict.py \
--data_dir data \
--test_data_file test.csv \
--model_type roberta \
--model_name RobertaBaseForSequenceClassification \
--model_path exps/checkpoint-2000 \
--max_length 128 \
--per_device_eval_batch_size 16 \
--submission_data_file sample_submission.csv \
--save_dir results \
--save_file_name best_public_score.csv