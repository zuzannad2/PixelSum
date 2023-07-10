from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from pixel import PoolingMode

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
    train_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze updating the weights of the decoder."}
    )

    def __post_init__(self):
        self.pooling_mode = PoolingMode.from_string(self.pooling_mode)

        if not self.rendering_backend.lower() in ["pygame", "pangocairo"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'pygame' and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()