from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from pixel import PoolingMode

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=529,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=60,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    log_predictions: bool = field(
        default=True,
        metadata={
            "help": "Whether to log predicted summaries."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    early_stopping_patience: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Number of epochs to wait for early stopping. Defaults to 5."
            )
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where to cache the datasets."
            )
        },
    )
    

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    predict_with_generate: Optional[bool] = field(
        default=True, metadata={"help": "Whether to generate text when running predictions."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets."
        }
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8,
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata = {"help": "Initial learning rate (after the potential warmup period) to use."}
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to push model to hub."}
    )
    hub_model_id: str = field(
        default=None
    )
    with_tracking: str = field(
        default=True
    )
    resume_from_checkpoint: str = field(
        default=None
    )
    checkpointing_steps: str = field(
        default=None
    )
    trust_remote_code: bool = field(
        default=True
    )
    hub_token: str = field(
        default=None
    )
    per_device_train_batch_size: int = field(
        default=32
    )
    num_train_epochs: int = field(
        default=10
    )
    weight_decay: float = field(
        default=0.0
    )
    max_train_steps: int = field(
        default=None
    )
    gradient_accumulation_steps: int = field(
        default=None
    )
    lr_scheduler_type: str = field(
        default=None
    )
    num_warmup_steps: int = field(
        default = 0
    )
    output_dir: str = field(
        default=None
    )
    seed: int = field(
        default=42
    )
    model_type: str = field(
        default='PIXELSum'
    )
    report_to: str = field(
        default='wandb'
    )
    logging_steps: int = field(
        default=1000
    )
    



 
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    encoder_name: str = field(
        metadata={"help": "Path to pretrained encoder or model identifier from huggingface.co/models"}
    )
    decoder_name: str = field(
        metadata={"help": "Path to pretrained decoder or model identifier from huggingface.co/models"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as encoder_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as decoder_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL. "
                          "PyGame does not support fallback fonts so this argument is ignored when using the "
                          "PyGame backend."},
    ) 
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
            "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
            "when there is need for it. PyGame does not support fallback fonts so this argument is ignored "
            "when using the PyGame backend."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": f"Pooling mode to use in classification head (options are {[e.value for e in PoolingMode]}."
        },
    )
    pooler_add_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "Whether to add layer normalization to the classification head pooler. Note that this flag is"
            "ignored and no layer norm is added when using CLS pooling mode."
        },
    )
    dropout_prob: float = field(
        default=0.2, metadata={"help": "Dropout probability for attention blocks and classification head"}
    )
    train_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to enable updating the weights of the encoder."}
    )
    train_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to enable updating the weights of the decoder. Note: XA always enabled."}
    )
    model_path: str = field(
        default=None,
        metadata={"help": (
            "Something"
        )}
    )

    def __post_init__(self):
        self.pooling_mode = PoolingMode.from_string(self.pooling_mode)

        if not self.rendering_backend.lower() in ["pygame", "pangocairo", "bigrams"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'bigrams', 'pygame', and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()