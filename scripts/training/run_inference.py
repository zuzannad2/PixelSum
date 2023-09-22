import logging
import os
import sys

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import argparse
from src.pixelsum.configuration_pixelsum import PIXELSumConfig
from src.pixelsum.modeling_pixelsum import PIXELSumModel

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
)
from pixel import (
    PangoCairoTextRenderer,
    PyGameTextRenderer,
    get_transforms,
)
from pixel.utils.misc import get_attention_mask


from schemas.custom_args import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

wandb.init(project="pixelsum")

def get_renderer(model_args: argparse.Namespace):
    renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
    renderer = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
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
        for pred, id in zip(preds, label_ids):
            p, r = tokenizer.decode(pred), tokenizer.decode(id)
            data.append([p, r])
            f.write(f"'Predicted: {p}\t, Reference: {r}\n")
            f.write("\n")

    logger.info(f"Saved predictions and labels to {out_file}")
    logger.info(f"Logging as table to wandb")

    preds_table = wandb.Table(columns=["Prediction", "Reference"], data=data)
    wandb.log({f"{prefix}_outputs": preds_table})


def load_model(model_args):  
    config = PIXELSumConfig.from_pretrained(
        "/home/vpz558/PixelSum/experiments/pretrained_gptsmall",
    )
    model = PIXELSumModel.from_pretrained(
            model_args.model_path,
            config=config
        )
    # model = PIXELSumModel.from_encoder_decoder_pretrained(
    #         model_args.encoder_name,
    #         model_args.decoder_name,
    #         cross_attention_reduce_factor=1
    #     )
    
    return model, model.config


def main():  
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))  
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
    
    # Load pretrained model
    model, config = load_model(model_args)
    renderer = get_renderer(model_args)
    tokenizer = get_tokenizer(model_args)
    
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.encoder.config.do_eval = True
    
    if 'gpt2' in model_args.decoder_name:
        model.decoder.config.task_specific_params['text-generation']['max_length'] = data_args.val_max_target_length
    
    transforms = get_transforms(
        do_resize=True, 
        size=(renderer.pixels_per_patch, renderer.pixels_per_patch * renderer.max_seq_length))

    def push_predictions_to_wandb(decoded_preds, decoded_labels, prefix):
        data = []
        out_file = os.path.join(training_args.output_dir, f"{prefix}_predictions.csv")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("Predicted\Reference\n")
            for p, r in zip(decoded_preds, decoded_labels):
                data.append([p, r])
                f.write(f"'Predicted: {p} \t, Reference: {r} \n")
                f.write("\n")

        logger.info(f"Saved predictions, masks and labels to {out_file}")
        logger.info(f"Logging as table to wandb")

        preds_table = wandb.Table(columns=["Predicted", "Reference"], data=data)
        wandb.log({f"{prefix}_outputs": preds_table})
    
    def push_metrics_to_wandb(metrics):
        metrics_table = wandb.Table(columns=list(metrics.keys()), data=[list(metrics.values())])
        wandb.log({"test_metrics": metrics_table})

    def preprocess_examples(batch):
        docs, summaries = batch['document'], batch['summary']
        data = {"pixel_values": [], "attention_mask": [], "label_ids": []}

        def _pad_input_ids(input_ids, max_length=data_args.max_target_length):
            input_ids += [-100] * (max_length - len(input_ids))
            return input_ids
        
        for document, summary in zip(docs, summaries):
            encoding = renderer(document)
            image = encoding.pixel_values
            num_patches = encoding.num_text_patches
            
            pixel_values = transforms(Image.fromarray(image))
            attention_mask = get_attention_mask(num_patches, seq_length=529)

            text_ids = tokenizer.encode(summary)
            text_ids = text_ids[:data_args.max_target_length] # Truncate
            input_ids = _pad_input_ids(text_ids) # Pad
        
            assert len(attention_mask) == 529
            
            data["pixel_values"].append(torch.tensor(pixel_values))
            data["attention_mask"].append(torch.tensor(attention_mask))
            data["label_ids"].append(torch.tensor(input_ids))

        return data
        
        
    test_dataset = load_dataset(data_args.dataset_name, split="test",cache_dir='cached_data')

    test_dataset = test_dataset.map(preprocess_examples, batched=True, remove_columns=["document", "summary", "id"],)
    
    if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))
            
    logger.warning(
        "Device(s): %s, n_gpu: %s, 16-bits training: %s",
        training_args.device,
        training_args.n_gpu,
        training_args.fp16,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def process_predictions(p: EvalPrediction):
        preds, labels = p
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = []
        decoded_labels = []
        for pred, label in zip(preds, labels):
            pred = np.argmax(pred, axis=1)
            decoded_pred = tokenizer.decode(pred)
                # Replace -100 in the labels as we can't decode them.
            label = np.where(label != -100, label, tokenizer.pad_token_id)
            decoded_label = tokenizer.decode(label, skip_special_tokens=True)
            decoded_preds.append(decoded_pred)
            decoded_labels.append(decoded_label)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        push_predictions_to_wandb(decoded_preds, decoded_labels, "test")
        return decoded_preds, decoded_labels

    bertscore, rouge = evaluate.load('bertscore'), evaluate.load('rouge')

    def compute_metrics(p: EvalPrediction):
        predictions, labels = process_predictions(p)
        rouge_res = rouge.compute(predictions=predictions, references=labels)
        bert = bertscore.compute(predictions=predictions, references=labels, lang='eng')
        bert_res = {'precision': np.mean(bert['precision']), 'recall': np.mean(bert['recall']), 'f1': np.mean(bert['f1'])}
        
        metrics = {
            "bertscore_precision": bert_res['precision'],
            "bertscore_recall": bert_res['recall'],
            "bertscore_f1": bert_res['f1'],
            "rouge1": rouge_res['rouge1'],
            "rouge2": rouge_res['rouge2'],
            "rougeL": rouge_res['rougeL'],
            "rougeLsum": rouge_res['rougeLsum']
        }

        push_metrics_to_wandb(metrics)
        return metrics

    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    training_args.generation_max_length = data_args.val_max_target_length

    logger.info("Training/evaluation parameters %s", training_args)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator, 
        train_dataset=None,
        eval_dataset=None,
        tokenizer=renderer,
        compute_metrics=compute_metrics,
    )

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