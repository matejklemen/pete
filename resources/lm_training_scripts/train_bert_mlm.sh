# resume_from_checkpoint=... # The path to a folder with a valid checkpoint for your model.
python3 run_mlm.py \
--model_type="bert" \
--model_name_or_path="bert-base-uncased" \
--use_fast_tokenizer \
--train_file="/home/matej/Documents/data/SST-2/lm_data/sst2_train_lm.txt" \
--validation_file="/home/matej/Documents/data/SST-2/lm_data/sst2_dev_lm.txt" \
--max_seq_length=44 \
--line_by_line \
--pad_to_max_length \
--output_dir="joze" \
--do_train \
--do_eval \
--logging_strategy="steps" \
--evaluation_strategy="steps" \
--save_strategy="steps" \
--logging_steps=1000 \
--save_steps=1000 \
--save_total_limit=7 \
--per_device_train_batch_size=32 \
--num_train_epochs=20 \
--report_to none

