# Training & Evaluation
python train.py \
--do_eval \
--PLM klue/roberta-large \
--validation_ratio 0.2 \
--max_length 128 \
--output_dir exps \
--data_dir data \
--train_data_file train.csv \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--eval_steps 400 \
--logging_steps 100 \
--evaluation_strategy steps \
--save_strategy no \
--learning_rate 2e-5 \
--warmup_ratio 0.05 \
--weight_decay 1e-3 \
--minimum_size 100 \
--maximum_size 3000

# Full Training
python train.py \
--PLM klue/roberta-large \
--max_length 128 \
--output_dir exps \
--data_dir data \
--train_data_file train.csv \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--save_steps 400 \
--logging_steps 100 \
--evaluation_strategy no \
--save_strategy steps \
--learning_rate 3e-5 \
--warmup_ratio 0.05 \
--weight_decay 1e-3

# Predict
python predict.py \
--data_dir data \
--test_data_file test.csv \
--model_path exps/checkpoint-2400 \
--max_length 128 \
--per_device_eval_batch_size 16 \
--submission_data_file sample_submission.csv \
--save_dir results \
--save_file_name 2022-12-16-01.csv