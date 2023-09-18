from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from pixel import PoolingMode

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
    per_device_train_batch_size: int 
    per_device_eval_batch_size: int = field(
        required=False,
        default=8,
    )
    learning_rate: float = field(
        required=True,
        default=5e-5,
        metadata = {"help": "Initial learning rate (after the potential warmup period) to use."}
    )
    num_train_epochs: int
    weight_decay: float 
    max_train_steps: int
    gradient_accumulation_steps: int
    lr_scheduler_type: str
    num_warmup_steps: int
    output_dir: str
    seed: int
    model_type: str
    report_to: str
    push_to_hub: bool = field(
        required=False,
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
    checkpointing_steps: int = field(
        default=None
    )
    trust_remote_code: bool = field(
        default=True
    )
    hub_token: str = field(
        default=None
    )



 
