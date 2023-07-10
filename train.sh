export ENCODER="Team-PIXEL/pixel-base"
export DECODER="gpt2-medium"
export DATASET="xsum"
export EXPERIMENT_DIR="experiments/"`date +%Y-%m-%d_%H-%M-%S`

mkdir -p ${EXPERIMENT_DIR}

python3 -m scripts.training.run_summarisation \
                   --encoder_name ${ENCODER} \
                   --decoder_name ${DECODER} \
                   --processor_name ${ENCODER} \
                   --tokenizer_name ${DECODER} \
                   --fallback_fonts_dir "fonts" \
                   --dataset_name ${DATASET} \
                   --dataloader_num_workers 16 \
                   --do_train true \
                   --do_eval true \
                   --do_predict true \
                   --train_decoder false \
                   --evaluation_strategy "steps" \
                   --eval_steps 600 \
                   --predict_with_generate true \
                   --logging_strategy "steps" \
                   --logging_steps 50 \
                   --save_strategy "steps" \
                   --save_steps 600 \
                   --fp16 true \
                   --fp16_full_eval true \
                   --output_dir ${EXPERIMENT_DIR} \
                   --overwrite_output_dir false \
                   --log_predictions false \
                   --per_device_train_batch_size 8 \
                   --per_device_eval_batch_size 1 \
                   --gradient_accumulation_steps 4 \
                   --eval_accumulation_steps 16 \
                   --learning_rate 0.001 \
                   --num_train_epochs 20 \
                   --val_max_target_length 150 \
                   --load_best_model_at_end true \
                   --metric_for_best_model 'rouge1' \
                   --use_fast_tokenizer true \
                   --max_eval_samples 7500 \
                   --num_beams 2 \

