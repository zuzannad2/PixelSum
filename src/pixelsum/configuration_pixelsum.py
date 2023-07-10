# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoConfig, PretrainedConfig, logging
import pixel
import copy

logger = logging.get_logger(__name__)

class PIXELSumConfig(PretrainedConfig):
    model_type = 'pixelsum'
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because "
                f"not both `encoder` and `decoder` sub-configurations are passed, but only {kwargs}"
            )

        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        self.encoder = pixel.AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True


    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`VisionEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`VisionEncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
