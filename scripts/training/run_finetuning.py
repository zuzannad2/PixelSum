import os

rank = int(os.getenv('RANK', '0'))
if rank == 0:
    import tensorflow as tf
    # Restrict TensorFlow to only use the last GPU. SMART-BLEURT will use (only) this GPU.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # This is hacky...
            tf.config.set_visible_devices(gpus[-1], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            assert len(logical_devices) == 1, "Only one GPU should be visible to TensorFlow"
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)
    # Also, only first process loads the SMART scores
    from smart_eval.scorer import SmartScorer
    from smart_eval.matching_functions import BleurtMatchingFunction, ChrfMatchingFunction
    
import logging
import sys

import datasets
import evaluate
import numpy as np
import pandas as pd 
from datasets import load_dataset, concatenate_datasets
from datetime import datetime

import argparse
from pixelsum.xglm import ThisXGLMConfig, ThisXGLMForCausalLM
from src.pixelsum.modeling_pixelsum import PIXELSumModel

import transformers
import torch
import torch.distributed as dist
from PIL import Image
import wandb
import copy
from tqdm import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    default_data_collator,
    EvalPrediction,
    HfArgumentParser,
    EarlyStoppingCallback,
    AutoConfig,
    AutoModel, 
    AutoModelForCausalLM,
    set_seed,
)
from pixel import (
    PangoCairoTextRenderer,
    PyGameTextRenderer,
    get_transforms,
)
from pixel.utils.misc import get_attention_mask
from src.pixelsum.modeling_pixelsum import PIXELSumModel, ThisSeq2SeqTrainer
from schemas.custom_args import ModelArguments, DataTrainingArguments, ThisSeq2SeqTrainingArguments

from src.pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsIsoCharRenderer
from src.pixel.data.rendering.pangocairo_renderer_sliding_window_bigrams import PangoCairoTextRenderer as PangoCairoSlidingWindowBigramsRenderer

logger = logging.getLogger(__name__)
     
wandb.init(project="pixelsum")

def get_encoder_processor(model_args: argparse.Namespace): 
    if "bert" in model_args.encoder_name:
        logger.info("Loading the (m)BERT tokenizer")
        return AutoTokenizer.from_pretrained(
            model_args.encoder_name,
            use_fast=True,
            add_prefix_space=True if model_args.encoder_name == "roberta-base" else False,
            cache_dir=model_args.cache_dir,
        )
    elif model_args.rendering_backend == "pygame":
        logger.info("Loading the PyGame renderer")
        processor_cls = PyGameTextRenderer 
    elif model_args.rendering_backend == "sliding_window_bigrams":
        logger.info("Loading the sliding window bigrams renderer")
        processor_cls = PangoCairoSlidingWindowBigramsRenderer 
    elif model_args.rendering_backend == "bigrams":
        logger.info("Loading the within-words bigrams renderer")
        processor_cls = PangoCairoBigramsIsoCharRenderer 
    else:
        logger.info("Loading the continuous renderer")
        processor_cls = PangoCairoTextRenderer 
        
    processor = processor_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
            use_auth_token=model_args.use_auth_token,
        )
    
    return processor

def get_decoder_tokenizer(model_args: argparse.Namespace): 
    tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.decoder_name,
            use_fast=model_args.use_fast_tokenizer,
            add_prefix_space=True if model_args.decoder_name == "gpt2" else False,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    tokenizer.pad_token = '<|pad|>'

    return tokenizer


def log_predictions(args, p, tokenizer, prefix):
    # Initialize wandb if not already done
    if not args.do_train:
        wandb.init(reinit=False)
    
    data = []
    out_file = os.path.join(args.output_dir, f"{prefix}_predictions.csv")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("summary\tpred\n")
        preds = np.argmax(p.predictions[0], axis=2)
        label_ids = p.label_ids
        for pred, id in zip(preds, label_ids):
            p, r = tokenizer.decode(pred), tokenizer.decode(id)
            data.append([p, r])
            f.write(f"'Pred: {p}\t, Sum: {r}\n")
            f.write("\n")

    logger.info(f"Saved predictions and labels to {out_file}")
    logger.info(f"Logging as table to wandb")

    preds_table = wandb.Table(columns=["summary", "pred"], data=data)
    wandb.log({f"{prefix}_outputs": preds_table})


def get_model_and_config(model_args: argparse.Namespace): 
    if "xglm" in model_args.decoder_name:
        AutoConfig.register("this_xglm", ThisXGLMConfig)
        AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
        AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    if len(model_args.model_path) == 0 or model_args.model_path is None:
        logger.info(f"Loading model from {model_args.encoder_name} and {model_args.decoder_name}")
        model = PIXELSumModel.from_encoder_decoder_pretrained(
            model_args.encoder_name,
            model_args.decoder_name,
            cross_attention_reduce_factor=1,
            training_loss_repetition_penalty=model_args.training_loss_repetition_penalty,
            use_auth_token=model_args.use_auth_token,
        )
    else:
        logger.info(f"Loading model from {model_args.model_path}")
        model = PIXELSumModel.from_pretrained(
                model_args.model_path,
            )
    
    if model_args.train_encoder:
        logger.info("Updating the entire encoder")
        pass
    elif "half" in model_args.train_encoder:
        logger.info("Only updating the top half of the encoder")
        for name, param in model.encoder.named_parameters():
            if "embeddings." in name:
                # we still want to train the patch embeddings (right?) 
                param.requires_grad_(True)
            elif any([f"layer.{n}." in name for n in range(0,6)]):
                # Disable the lower layers
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
    else:
        logger.info("Freezing the entire encoder (except the patch embedding layer)")
        for name, param in model.encoder.named_parameters():
            if "embeddings." in name:
                # we still want to train the patch embeddings (right?) 
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    
    if "opt" in model_args.decoder_name or "xglm" in model_args.decoder_name or "bloom" in model_args.decoder_name:
        if not model_args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'encoder_attn' not in name:
                    param.requires_grad_(False)

    elif "gpt" in model_args.decoder_name:
        if not model_args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    param.requires_grad_(False)
   
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    
    logger.info('Training a model with {} trainable parameters.'.format(num_trainable_params))
    logger.info(f"Using dropout with probability {model_args.dropout_prob}")
    
    return model, model.config

def main():  
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ThisSeq2SeqTrainingArguments))  
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Initialise model and processors
    model, config = get_model_and_config(model_args)
    processor = get_encoder_processor(model_args)
    tokenizer = get_decoder_tokenizer(model_args)
    logger.info(f"{tokenizer.is_fast=}")
    logger.info(f"Process {rank=}; GPU rank={torch.cuda.current_device()}") 
    
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.encoder.config.do_eval = True

    if 'gpt2' in model_args.decoder_name:
        model.decoder.config.task_specific_params['text-generation']['max_length'] = data_args.val_max_target_length
    
    if not "bert" in model_args.encoder_name:
        transforms = get_transforms(
            do_resize=True, 
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length))
    
    def preprocess_examples(batch):
        documents, summaries = batch['text'], batch['target']
        data = {"pixel_values": [], "attention_mask": [], "label_ids": []}
        
        def _pad_input_ids(input_ids, max_length=data_args.max_target_length):
            input_ids += [-100] * (max_length - len(input_ids))
            return input_ids
        
        for document, summary in zip(documents, summaries):
            # NOTE consider getting word boundaries from Stanza for Chinese (and other languages?)
            document = document.replace('\n','')
            encoding = processor(document)

            image = encoding.pixel_values
            num_patches = encoding.num_text_patches
            pixel_values = transforms(Image.fromarray(image))
            attention_mask = get_attention_mask(num_patches, seq_length=data_args.max_seq_length)
            
            text_ids = tokenizer.encode(summary, add_special_tokens=False)
            text_ids = text_ids[:data_args.max_target_length] # Truncate
            input_ids = _pad_input_ids(text_ids) # Pad
       
            assert len(attention_mask) == data_args.max_seq_length
            
            data["pixel_values"].append(pixel_values)
            data["attention_mask"].append(attention_mask)
            data["label_ids"].append(torch.tensor(input_ids))

        return data

    def preprocess_examples_bert(batch):
        documents, summaries = batch['text'], batch['target']
        data = {"pixel_values": [], "attention_mask": [], "label_ids": []}
        
        def _pad_input_ids(input_ids, max_length=data_args.max_target_length):
            input_ids += [-100] * (max_length - len(input_ids))
            return input_ids
        
        for document, summary in zip(documents, summaries):
            document = document.replace('\n','')
            _data = processor(document, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

            text_ids = tokenizer.encode(summary, add_special_tokens=False)
            text_ids = text_ids[:data_args.max_target_length] # Truncate
            input_ids = _pad_input_ids(text_ids) # Pad
                   
            data["pixel_values"].append(_data['input_ids']) # NOTE not so pretty hack. Fix later !!!
            data["attention_mask"].append(_data['attention_mask'])
            data["label_ids"].append(torch.tensor(input_ids))

        return data

    dataset = load_dataset(data_args.dataset_name, data_args.language, keep_in_memory=True)

    if training_args.do_train:
        train_dataset = dataset['train']
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if "bert" in model_args.encoder_name:
            logger.info("Preprocessing examples for BERT")
            train_dataset.set_transform(preprocess_examples_bert)
        else:
            train_dataset.set_transform(preprocess_examples)
        logger.info(f'Successfully loaded the training data with {len(train_dataset)} examples.')

    if training_args.do_eval:
        val_dataset = dataset['validation']
        if data_args.max_eval_samples is not None:
            val_dataset = val_dataset.select(range(data_args.max_eval_samples)).shuffle(seed=training_args.seed)
        copy_val_dataset_text = val_dataset['text'].copy()
        if "bert" in model_args.encoder_name:
            val_dataset.set_transform(preprocess_examples_bert)
        else:
            val_dataset.set_transform(preprocess_examples)
        logger.info(f'Successfully loaded the validation data with {len(val_dataset)} examples.')

    if training_args.do_predict:
        test_dataset = dataset['test']
        if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))
        copy_test_dataset = copy.deepcopy(test_dataset)
        if "bert" in model_args.encoder_name:
            test_dataset.set_transform(preprocess_examples_bert)
        else:
            test_dataset.set_transform(preprocess_examples)
        logger.info(f'Successfully loaded the testing data with {len(test_dataset)} examples.')

    logger.warning(
        "Device: %s, n_gpu: %s, (fp)16-bits training: %s",
        training_args.device,
        training_args.n_gpu,
        training_args.fp16,
    )

    def push_predictions_to_wandb(decoded_preds, decoded_labels):
        data = [[p,r] for p,r in zip(decoded_preds, decoded_labels)]
        wandb.log({f"training_outputs": wandb.Table(columns=["Prediction", "Summary"], data=data)})

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    def process_predictions(p: EvalPrediction):
        preds, labels = p
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        push_predictions_to_wandb(decoded_preds[0:10], decoded_labels[0:10])
        return decoded_preds, decoded_labels

    # bertscore = evaluate.load('bertscore',) # Not reporting bertscore
    rouge = evaluate.load('rouge')
    if rank == 0: 
        # As at the top of the docment, only the first process, which has access to the last GPU, 
        # loads and computes the SMART scores
        bleurt_scorer = SmartScorer(matching_fn=BleurtMatchingFunction(
            data_args.bleurt_model_path,
        ))
        chrf_scorer = SmartScorer(matching_fn=ChrfMatchingFunction())

    def compute_metrics(p: EvalPrediction):
        predictions, labels = process_predictions(p)
        rouge_res = rouge.compute(predictions=predictions, references=labels)
        
        if rank == 0: 
            # assert len(predictions) == len(copy_val_dataset_text), "Number of predictions and number of examples do not match"
            smart_eval_chrf_list = []
            smart_eval_bleurt_list = []
            # Compute SMART scores for each example 
            for pred, label, src in tqdm(zip(predictions, labels, copy_val_dataset_text), 
                                        total=len(predictions), desc="Computing SMART scores", leave=False):
                smart_eval_chrf_list.append(chrf_scorer.smart_score(
                    candidate=pred, 
                    reference=label,
                    source=src)['smartL']['fmeasure']) # We only care about smartL
                smart_eval_bleurt_list.append(bleurt_scorer.smart_score(
                    candidate=pred, 
                    reference=label,
                    source=src)['smartL']['fmeasure'])
            
            smart_eval_chrf_res = torch.tensor([np.mean(smart_eval_chrf_list)], 
                                               device=torch.device("cuda:0"), dtype=torch.float32) 
            # NOTE it's important to specify device and dtype!
            smart_eval_bleurt_res = torch.tensor([np.mean(smart_eval_bleurt_list)], 
                                                 device=torch.device("cuda:0"), dtype=torch.float32)
            logger.info(f"rank == 0: {smart_eval_chrf_res=}; {smart_eval_bleurt_res=}")
            
        else: # For the other processes, create placeholders for the results
            smart_eval_chrf_res = torch.zeros([1], 
                                              device=torch.device(f"cuda:{rank}"), dtype=torch.float32)
            # NOTE dtype needs to be the same as for the main process
            smart_eval_bleurt_res = torch.zeros([1], 
                                                device=torch.device(f"cuda:{rank}"), dtype=torch.float32)
            logger.info(f"rank != 0: {smart_eval_chrf_res=}; {smart_eval_bleurt_res=}")
        
        dist.barrier() # shouldn't be necessary
        dist.broadcast(smart_eval_chrf_res, src=0)
        dist.broadcast(smart_eval_bleurt_res, src=0)
        smart_eval_chrf_out = smart_eval_chrf_res.item()
        smart_eval_bleurt_out = smart_eval_bleurt_res.item()
        logger.info(f"After broadcasting ({rank=}): {smart_eval_chrf_out=}; {smart_eval_bleurt_out=}")
      
        # Save predictions as csv incl. timestamp
        if data_args.log_predictions:
            logger.info("Saving predictions to csv")
            df_out = pd.DataFrame({'predictions': predictions, 'labels': labels, "source": copy_val_dataset_text})
            out_file = os.path.join(training_args.output_dir, f"predictions_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv")
            df_out.to_csv(out_file, index=False, sep='\t', header=True)
    
        return {
            "rouge1": rouge_res['rouge1'],
            "rouge2": rouge_res['rouge2'],
            "rougeL": rouge_res['rougeL'],
            "rougeLsum": rouge_res['rougeLsum'],
            "SMART-CHRF_L-Fmeasure": smart_eval_chrf_out,
            "SMART-BLEURT_L-Fmeasure": smart_eval_bleurt_out,
        }

    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    model.decoder.config.num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    training_args.early_stopping_patience = (
        data_args.early_stopping_patience if data_args.early_stopping_patience is not None else training_args.early_stopping_patience
    )

    training_args.generation_max_length = data_args.val_max_target_length

    logger.info("Training/evaluation parameters %s", training_args)

    trainer = ThisSeq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator, 
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        #callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
    )

    last_checkpoint = None

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(test_dataset, metric_key_prefix="test") 
        
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        
        if data_args.log_predictions:    
            out_file = os.path.join(training_args.output_dir, "test_predictions.csv")
            preds, labels = predict_results.predictions, predict_results.label_ids
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            processed_preds, processed_labels = postprocess_text(decoded_preds, decoded_labels)
            df = pd.DataFrame(copy_test_dataset)
            df['predictions'] = pd.Series(processed_preds)
            df['processed_labels'] = pd.Series(processed_labels)
            df.to_csv(out_file, index=False, sep='\t', header=True)

            logger.info(f"Saved predictions and labels to {out_file}")

    
if __name__ == '__main__':
    main()