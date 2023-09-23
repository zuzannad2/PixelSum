# Setup 

This repository builts upon [**PIXEL**](https://github.com/xplip/pixel) and extends the linked repository with a summarisation model based on the **Pix**el-based **E**ncoder of **L**anguage. This codebase is built on [Transformers](https://github.com/huggingface/transformers) for PyTorch. The default font `GoNotoCurrent.ttf` that we used for all experiments is a merged Noto font built with [go-noto-universal](https://github.com/satbyy/go-noto-universal). 

You can set-up this codebase in the following way:

<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;

1. Clone repo and initialize submodules
```
git clone https://github.com/zuzannad2/PixelSum.git
cd PixelSum
git submodule update --init --recursive
```

2. Create a fresh conda environment
```
conda create -n venv python=3.9
conda activate venv
```

3. Install Python packages
```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -c conda-forge pycairo pygobject manimpango
pip install --upgrade pip
pip install -r requirements.txt
pip install ./datasets
pip install -e .
```

4. Download fallback fonts using ```python3 -m scripts.data.download_fallback_fonts fonts```. Download the font GoNotoCurrent.ttf using the repo [go-noto-universal](https://github.com/satbyy/go-noto-universal) into the folder called "fonts".

## Pre-training PixelSum
The pretraining scripts for training PixelSum with Huggingface's trainer (```run_pretraining.py```) or without the trainer (```run_pretrainer_no_trainer.py```) are located in ```scripts/training```. These are meant to be ran via bash scripts ```pretrain.sh``` / ```pretrain_no_trainer.sh``` and are configured for Slurm. 

## Finetuning PixelSum
The finetuning script for PixelSum is in ```scripts/training/run_finetuning.py```. The script is ran via ```finetune.sh```. 
- You need to pass the pretrained model path in the  ```model_path``` argument.
- The arguments ```train_encoder``` and ```train_decoder``` are set to False, meaning by default only the cross-attention layers will be trained. Adjust to desired setting. 

## Inference 
The inference script is in ```scripts/training/run_inference.py```. The script is ran via ```infer.sh```. 
- Pass the pretrained model in ```model_path``` argument (leave empty string if zero-shot inference from model initialised ```from_encoder_decoder_pretrained```).

## Logging
By default scripts are set to log to WandB.