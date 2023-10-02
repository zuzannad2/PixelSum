wandb online

export PYTHONPATH="/home/rff/Documents/PixelSum"

export WANDB_PROJECT="pixelsum"
export WANDB_ENTITY="zuzannad"
export WANDB_API_KEY="05ddd851adccb0f9c0393bcaa28794de1ad7c370"

export ENCODER="Team-PIXEL/pixel-base"
export DECODER="facebook/opt-350m"
# export DECODER="gpt2"
export DATASET="zuzannad1/pixelsum_wiki"
export EXPERIMENT_DIR="experiments/pretraining/$DECODER/"`date +%Y-%m-%d_%H-%M-%S`

mkdir -p ${EXPERIMENT_DIR}

python /home/rff/Documents/PixelSum/scripts/training/run_pretraining.py \
    --encoder_name=${ENCODER} \
    --decoder_name=${DECODER} \
    --processor_name=${ENCODER} \
    --tokenizer_name=${DECODER} \
    --fallback_fonts_dir="fonts" \
    --rendering_backend="pangocairo" \
    --dataset_name=${DATASET} \
    --remove_unused_columns=False \
    --dataloader_num_workers=1 \
    --do_train=True \
    --do_eval=True \
    --do_predict=True \
    --train_decoder=True \
    --train_encoder=True \
    --evaluation_strategy="steps" \
    --eval_steps=10000 \
    --predict_with_generate=True \
    --logging_strategy="steps" \
    --logging_steps=50 \
    --save_strategy="steps" \
    --save_steps=10000 \
    --fp16=True \
    --output_dir=${EXPERIMENT_DIR} \
    --overwrite_output_dir=False \
    --log_predictions=True \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1.5e-4 \
    --weight_decay=0.05 \
    --warmup_ratio=0.05 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=15000 \
    --max_steps=300000 \
    --val_max_target_length=50 \
    --max_target_length=50 \
    --load_best_model_at_end=True \
    --use_fast_tokenizer=True \
    --num_beams=1 \
    --report_to="wandb" \
    --max_eval_samples=100 \
    --torch_compile \
    --data_cache_dir='cached_data' 
                   