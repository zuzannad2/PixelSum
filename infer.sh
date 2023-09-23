#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --job-name=infer-pixelsum
#SBATCH --ntasks=1 --cpus-per-task=48 --mem=70000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=3:00:00

nvidia-smi

export ENCODER="Team-PIXEL/pixel-base"
export DECODER="gpt2"
export DATASET="xsum"
export EXPERIMENT_DIR="experiments/inference/$DECODER/"`date +%Y-%m-%d_%H-%M-%S`

mkdir -p ${EXPERIMENT_DIR}

python3 -m scripts.training.run_inference \
                --model_path "" \
                --encoder_name ${ENCODER} \
                --decoder_name ${DECODER} \
                --processor_name ${ENCODER} \
                --tokenizer_name ${DECODER} \
                --fallback_fonts_dir "fonts" \
                --dataset_name ${DATASET} \
                --dataloader_num_workers 32 \
                --do_train false \
                --do_eval false \
                --do_predict true \
                --train_decoder false \
                --train_encoder false \
                --output_dir ${EXPERIMENT_DIR} \
                --log_predictions true \
                --eval_accumulation_steps 8 \
                --per_device_eval_batch_size 1 \
                --num_beams 1 \
                --max_predict_samples 1500 \
                --data_cache_dir 'cached_data' \