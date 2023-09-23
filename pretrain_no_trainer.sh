#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --job-name=pretrain-pixel-gpt2large
#SBATCH --ntasks=1 --cpus-per-task=48 --mem=70000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=75:00:00

nvidia-smi

export ENCODER="Team-PIXEL/pixel-base"
export DECODER="gpt2-large"
export DATASET="zuzannad1/pixelsum_wiki"
export EXPERIMENT_DIR="experiments/pretraining/$DECODER/"`date +%Y-%m-%d_%H-%M-%S`

mkdir -p ${EXPERIMENT_DIR}

accelerate launch --mixed_precision=fp16 scripts/training/run_pretraining_no_trainer.py \
                   --encoder_name ${ENCODER} \
                   --decoder_name ${DECODER} \
                   --processor_name ${ENCODER} \
                   --tokenizer_name ${DECODER} \
                   --fallback_fonts_dir "fonts" \
                   --dataset_name ${DATASET} \
                   --train_decoder true \
                   --train_encoder true \
                   --predict_with_generate true \
                   --checkpointing_steps '10000' \
                   --output_dir ${EXPERIMENT_DIR} \
                   --log_predictions true \
                   --per_device_train_batch_size 32 \
                   --gradient_accumulation_steps 1 \
                   --learning_rate 1.5e-4 \
                   --weight_decay 0.05 \
                   --lr_scheduler_type "cosine" \
                   --num_warmup_steps 15000 \
                   --num_train_epochs 2 \
                   --val_max_target_length 50 \
                   --max_target_length 50 \
                   --use_fast_tokenizer true \
                   --num_beams 2 \
                   --report_to "wandb" \
                   --pad_to_max_length true \
                   --max_train_steps 300000 \
                   --logging_steps 50 \
                   --data_cache_dir 'cached_data' \
                  
                  