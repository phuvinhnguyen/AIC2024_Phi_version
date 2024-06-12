from transformers import PerceiverConfig, PretrainedConfig

class DoublePerceiverConfig(PretrainedConfig):
    model_type = "double_perceiver"

    def __init__(self,
                 video_config = {},
                 frame_config = {},
                 **kwargs
                 ):
        self.video_config = PerceiverConfig(**video_config)
        self.frame_config = PerceiverConfig(**frame_config)

        super().__init__(**kwargs)