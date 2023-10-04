#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --account dd-23-53
#SBATCH --job-name=ft-xaopt350m
#SBATCH --ntasks=1 --cpus-per-task=48 --mem=70000M
#SBATCH --partition qgpu
#SBATCH --time=10:00:00

cd $SLURM_SUBMIT_DIR
source /scratch/project/dd-23-53/installs/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/project/dd-23-53/env/venv

ml CUDA/11.7.0
ml GCC/8.3.0

export ENCODER="Team-PIXEL/pixel-base"
export DECODER="facebook/opt-1.3b"
export DATASET="xsum"
export EXPERIMENT_DIR="experiments/finetune/$DECODER/enc/"`date +%Y-%m-%d_%H-%M-%S`

mkdir -p ${EXPERIMENT_DIR}

python3 -m scripts.training.run_finetuning \
                   --model_path "" \
                   --encoder_name ${ENCODER} \
                   --decoder_name ${DECODER} \
                   --processor_name ${ENCODER} \
                   --tokenizer_name ${DECODER} \
                   --rendering_backend "pangocairo" \
                   --fallback_fonts_dir "fonts" \
                   --dataset_name ${DATASET} \
                   --dataloader_num_workers 1 \
                   --remove_unused_columns false \
                   --do_train true \
                   --do_eval false \
                   --do_predict true \
                   --train_decoder true \
                   --train_encoder true \
                   --evaluation_strategy "steps" \
                   --eval_steps 500 \
                   --predict_with_generate true \
                   --logging_strategy "steps" \
                   --logging_steps 50 \
                   --save_strategy "steps" \
                   --save_steps 1000 \
                   --fp16 true \
                   --fp16_full_eval false \
                   --output_dir ${EXPERIMENT_DIR} \
                   --overwrite_output_dir false \
                   --log_predictions true \
                   --per_device_train_batch_size 4 \
                   --gradient_accumulation_steps 16 \
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
                   --repetition_penalty=5.0 \
                   --data_cache_dir '/scratch/project/dd-23-53/zuz/' \
                   --torch_compile
                   


