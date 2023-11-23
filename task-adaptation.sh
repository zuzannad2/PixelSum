# cd /scratch/project/dd-23-53/PixelSum
# cd $SLURM_SUBMIT_DIR
# print pwd
echo $PWD

export PYTHONPATH="/home/jflotz/Documents/PixelSum"

# export ENCODER="Team-PIXEL/pixel-base-bigrams"
export ENCODER="bert-base-multilingual-cased"
export DECODER="facebook/xglm-1.7b"
export DATASET="GEM/xlsum"
export LANGUAGE="english"
export BLANG="en"
export EXPERIMENT_DIR="experiments/TA/$DECODER/enc/"`date +%Y-%m-%d_%H-%M-%S`

export HF_DATASETS_CACHE="/home/jflotz/Documents/PixelSum/cached_data/datasets"
export HF_METRICS_CACHE="/home/jflotz/Documents/PixelSum/cached_data/metrics"
export TRANSFORMERS_CACHE="/home/jflotz/Documents/PixelSum/cached_data/transformers"
export HF_MODULES_CACHE="/home/jflotz/Documents/PixelSum/cached_data/modules"
export WANDB_CACHE_DIR="/home/jflotz/Documents/PixelSum/cached_data/wandb"

mkdir -p ${EXPERIMENT_DIR}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --max_restarts 0 --standalone /home/jflotz/Documents/PixelSum/scripts/training/run_finetuning_zuz.py \
                   --model_path "" \
                   --encoder_name ${ENCODER} \
                   --decoder_name ${DECODER} \
                   --processor_name "Team-PIXEL/pixel-base" \
                   --tokenizer_name ${DECODER} \
                   --rendering_backend "bigrams" \
                   --fallback_fonts_dir "fonts" \
                   --dataset_name ${DATASET} \
                   --language ${LANGUAGE} \
                   --bert_lang ${BLANG} \
                   --dataloader_num_workers 12 \
                   --remove_unused_columns false \
                   --do_train true \
                   --do_eval false \
                   --do_predict true \
                   --train_decoder false \
                   --train_encoder true \
                   --evaluation_strategy "steps" \
                   --eval_steps 500 \
                   --predict_with_generate true \
                   --logging_strategy "steps" \
                   --logging_steps 50 \
                   --save_strategy "steps" \
                   --save_steps 1000 \
                   --bf16 \
                   --fp16_full_eval false \
                   --output_dir ${EXPERIMENT_DIR} \
                   --overwrite_output_dir false \
                   --log_predictions true \
                   --per_device_train_batch_size 1 \
                   --gradient_accumulation_steps 1 \
                   --learning_rate 3e-4 \
                   --weight_decay 0.05 \
                   --warmup_ratio 0.05 \
                   --lr_scheduler_type "cosine" \
                   --num_train_epochs 10 \
                   --val_max_target_length 50 \
                   --max_target_length 50 \
                   --load_best_model_at_end true \
                   --metric_for_best_model 'rouge1' \
                   --use_fast_tokenizer true \
                   --num_beams 3 \
                   --report_to "wandb" \
                   --max_eval_samples 500 \
                   --data_cache_dir '/home/jflotz/Documents/PixelSum/cached_data' \
                   --torch_compile true \
