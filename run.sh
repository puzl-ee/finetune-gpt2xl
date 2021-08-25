export VOLUME_DIR="/media/gpt-neo-experiment"
export TRANSFORMERS_CACHE="${VOLUME_DIR}/model_cache"
export MODEL_OUTPUT_DIR="${VOLUME_DIR}/finetuned_model"

pip3 install --user -r requirements.txt

deepspeed --num_gpus=1 run_clm.py \
  --deepspeed ds_config_gptneo.json \
  --model_name_or_path EleutherAI/gpt-neo-2.7B \
  --train_file train.csv \
  --validation_file validation.csv \
  --do_train \
  --do_eval \
  --fp16 \
  --overwrite_cache \
  --evaluation_strategy="steps" \
  --output_dir ${MODEL_OUTPUT_DIR} \
  --num_train_epochs 1 \
  --eval_steps 15 \
  --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 1 \
  --use_fast_tokenizer False \
  --learning_rate 5e-06 \
  --warmup_steps 10

python3 run_generate_neo.py ${MODEL_OUTPUT_DIR}
