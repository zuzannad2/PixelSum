import logging
import os
import sys

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import argparse
import transformers
import torch
import numpy as np
from PIL import Image
import wandb

import transformers
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    default_data_collator,
    Seq2SeqTrainingArguments,
    EvalPrediction,
    HfArgumentParser,
    EarlyStoppingCallback
)
from pixel import (
    PangoCairoTextRenderer,
    PyGameTextRenderer,
    get_transforms,
)
from pixel.utils.misc import get_attention_mask
from src.pixelsum.modeling_pixelsum import PIXELSumModel, ThisSeq2SeqTrainer
from schemas.custom_args import ModelArguments, DataTrainingArguments, ThisSeq2SeqTrainingArguments
from src.pixel.data.rendering.pangocairo_renderer_bigrams import PangoCairoTextRenderer as PangoCairoBigramsRenderer

logger = logging.getLogger(__name__)

wandb.init(project="pixelsum")

def get_renderer(model_args: argparse.Namespace):
    if model_args.rendering_backend == "pygame":
        renderer_cls = PyGameTextRenderer 
    elif model_args.rendering_backend == "bigrams":
        logger.info("Loading bigrams renderer")
        renderer_cls = PangoCairoBigramsRenderer 
    else:
        renderer_cls = PangoCairoTextRenderer 
        
    renderer = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
            use_auth_token=model_args.use_auth_token,
        )
    
    return renderer

def get_tokenizer(model_args: argparse.Namespace):
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
        f.write("Prediction\Reference\n")
        preds = np.argmax(p.predictions[0], axis=2)
        label_ids = p.label_ids
        for pred, id in zip(preds[:5], label_ids[:5]): # only log first 5
            p, r = tokenizer.decode(pred), tokenizer.decode(id)
            data.append([p, r])
            f.write(f"'Predicted: {p}\t, Reference: {r}\n")
            f.write("\n")

    logger.info(f"Saved predictions and labels to {out_file}")
    logger.info(f"Logging as table to wandb")

    preds_table = wandb.Table(columns=["Prediction", "Reference"], data=data)
    wandb.log({f"{prefix}_outputs": preds_table})


def get_model_and_config(model_args: argparse.Namespace):  

    model = PIXELSumModel.from_encoder_decoder_pretrained(
            model_args.encoder_name,
            model_args.decoder_name,
            cross_attention_reduce_factor=1,
            training_loss_repetition_penalty=model_args.training_loss_repetition_penalty,
            use_auth_token=model_args.use_auth_token,
        )
    if not model_args.train_encoder:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
    else:
        for name, param in model.encoder.named_parameters():
            if "embeddings." in name:
                # we don't want to train the patch embeddings
                param.requires_grad_(False)
            elif any([f"layer.{n}." in name for n in range(0,6)]):
                # Disable the lower layers
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
            # Disable the lower layers
            # for n in range(0,6): 
            #     if f"encoder.layer.{n}." in name:
            #         param.requires_grad_(False)

    if "opt" in model_args.decoder_name or "xglm" in model_args.decoder_name:
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
    renderer = get_renderer(model_args)
    tokenizer = get_tokenizer(model_args)
    
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 
    #model.encoder.config.do_eval = True
    
    if 'gpt2' in model_args.decoder_name:
        model.decoder.config.task_specific_params['text-generation']['max_length'] = data_args.val_max_target_length
    
    transforms = get_transforms(
        do_resize=True, 
        size=(renderer.pixels_per_patch, renderer.pixels_per_patch * renderer.max_seq_length))
        
    def string_to_ngrams(s:str, n:int=2) -> list:
        """
        Takes a string and returns a list of character n-grams by splitting `s` on every `n` character.
        Args:
            s (str): The input string to be converted to bigrams.
            n (int): The frequency of which the input string is split. Defaults to `n`=2
        Returns:
            list: A list of character n-grams.
        """
        return [s[i:i + n] for i in range(0, len(s), n)]

    def preprocess_examples(batch):
        docs, summaries = batch['example'], batch['summary']
        data = {"pixel_values": [], "attention_mask": [], "label_ids": []}

        def _pad_input_ids(input_ids, max_length=data_args.max_target_length):
            input_ids += [-100] * (max_length - len(input_ids))
            return input_ids
        
        for document, summary in zip(docs, summaries):
            if model_args.rendering_backend == "bigrams":
                encoding = renderer(string_to_ngrams(' '.join(document.split())))
            else:
                # continuous rendering of text
                encoding = renderer(document)
            image = encoding.pixel_values
            num_patches = encoding.num_text_patches
            
            pixel_values = transforms(Image.fromarray(image))
            attention_mask = get_attention_mask(num_patches, seq_length=529)

            text_ids = tokenizer.encode(summary, add_special_tokens=False) 
            # text_ids = tokenizer.encode(summary) 
            # input_ids = tokenizer.encode(summary, add_special_tokens=True, 
            #                             padding="max_length", 
            #                             truncation=True,
            #                             max_length=data_args.max_target_length)
            text_ids = text_ids[:data_args.max_target_length] # Truncate
            input_ids = _pad_input_ids(text_ids) # Pad
            # print(f"PROCESS EXAMPLES: {input_ids=}")
        
            assert len(attention_mask) == 529
            
            data["pixel_values"].append(torch.tensor(pixel_values))
            data["attention_mask"].append(torch.tensor(attention_mask))
            data["label_ids"].append(torch.tensor(input_ids))

        return data
        
        
    dataset = load_dataset(data_args.dataset_name, split="train",cache_dir=data_args.data_cache_dir) 
    
    split = dataset.train_test_split(test_size=0.1)
    
    if training_args.do_train:
        train_dataset = split['train']
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples)).shuffle(seed=42)
        train_dataset.set_transform(preprocess_examples)
        #train_dataset = train_dataset.map(preprocess_examples, batched=True, remove_columns=["example", "summary"],num_proc=data_args.preprocessing_num_workers)
        logger.info(f'Successfully loaded the training data with {len(train_dataset)} examples.')
    
    if training_args.do_eval:
        val_dataset = split['test']
        if data_args.max_eval_samples is not None:
            val_dataset = val_dataset.select(range(data_args.max_eval_samples))
        val_dataset.set_transform(preprocess_examples)
        #val_dataset = val_dataset.map(preprocess_examples, batched=True, remove_columns=["example", "summary"],num_proc=data_args.preprocessing_num_workers)
        logger.info(f'Successfully loaded the validation data with {len(val_dataset)} examples.')

    if training_args.do_predict:
        test_dataset = split['test']
        if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))
        test_dataset.set_transform(preprocess_examples)
        #test_dataset = test_dataset.map(preprocess_examples, batched=True, remove_columns=["example", "summary"],num_proc=data_args.preprocessing_num_workers)
        logger.info(f'Successfully loaded the testing data with {len(test_dataset)} examples.')

    logger.warning(
        "Device(s): %s, n_gpu: %s, 16-bits training: %s",
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
        # logger.info(f"{preds=}")
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

    bertscore, rouge = evaluate.load('bertscore'), evaluate.load('rouge')

    def compute_metrics(p: EvalPrediction):
        predictions, labels = process_predictions(p)
        rouge_res = rouge.compute(predictions=predictions, references=labels)
        bert = bertscore.compute(predictions=predictions, references=labels, lang='eng')
        bert_res = {'precision': np.mean(bert['precision']), 'recall': np.mean(bert['recall']), 'f1': np.mean(bert['f1'])}
        
        return {
            "bertscore_precision": bert_res['precision'],
            "bertscore_recall": bert_res['recall'],
            "bertscore_f1": bert_res['f1'],
            "rouge1": rouge_res['rouge1'],
            "rouge2": rouge_res['rouge2'],
            "rougeL": rouge_res['rougeL'],
            "rougeLsum": rouge_res['rougeLsum']
        }

    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    training_args.early_stopping_patience = (
        data_args.early_stopping_patience if data_args.early_stopping_patience is not None else training_args.early_stopping_patience
    )

    training_args.generation_max_length = data_args.val_max_target_length

    logger.info("Training/evaluation parameters %s", training_args)
    
    # trainer = Seq2SeqTrainer(
    trainer = ThisSeq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator, 
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=renderer,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
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
        
    
if __name__ == '__main__':
    main()
