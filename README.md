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

4. Download fallback fonts using ```python3 -m pixel.scripts.data.download_fallback_fonts.py fonts```. Download the font GoNotoCurrent.ttf using the repo [go-noto-universal](https://github.com/satbyy/go-noto-universal) into the folder called "fonts".

## Finetuning PixelSum
PixelSum can be finetuning by running ```train.sh``` which runs the ```run_summarisation.py``` script.