#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --job-name=finetune-pixelsmall
#SBATCH --ntasks=1 --cpus-per-task=48 --mem=70000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=20:00:00

export ENCODER="Team-PIXEL/pixel-base"
export DECODER="gpt2"
export DATASET="xsum"
export EXPERIMENT_DIR="experiments/finetune/$DECODER/"`date +%Y-%m-%d_%H-%M-%S`

mkdir -p ${EXPERIMENT_DIR}

python3 -m scripts.training.run_finetuning \
                   --model_path "/home/vpz558/PixelSum/experiments/pretrained_gptsmall" \
                   --encoder_name ${ENCODER} \
                   --decoder_name ${DECODER} \
                   --processor_name ${ENCODER} \
                   --tokenizer_name ${DECODER} \
                   --fallback_fonts_dir "fonts" \
                   --dataset_name ${DATASET} \
                   --dataloader_num_workers 32 \
                   --do_train true \
                   --do_eval false \
                   --do_predict true \
                   --train_decoder false \
                   --train_encoder false \
                   --evaluation_strategy "steps" \
                   --eval_steps 1000 \
                   --predict_with_generate true \
                   --logging_strategy "steps" \
                   --logging_steps 50 \
                   --save_strategy "steps" \
                   --save_steps 1000 \
                   --fp16 true \
                   --fp16_full_eval true \
                   --output_dir ${EXPERIMENT_DIR} \
                   --overwrite_output_dir false \
                   --log_predictions true \
                   --per_device_train_batch_size 64 \
                   --gradient_accumulation_steps 1 \
                   --learning_rate 1.5e-4 \
                   --weight_decay 0.05 \
                   --warmup_ratio 0.05 \
                   --lr_scheduler_type "cosine" \
                   --warmup_steps 1000 \
                   --max_steps 15000 \
                   --val_max_target_length 50 \
                   --max_target_length 50 \
                   --load_best_model_at_end true \
                   --use_fast_tokenizer true \
                   --num_beams 1 \
                   --report_to "wandb" \
                   --max_eval_samples 100 \
                   --max_predict_samples 1500 \
                   


