export TRAIN_FILE=train.txt
export TEST_FILE=test.txt

python run_clm.py \
    --output_dir=gpt-recipes \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --per_device_train_batch_size 4 \
    --train_file=$TRAIN_FILE \
    --do_eval \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --overwrite_output_dir \
    --validation_file=$TEST_FILE