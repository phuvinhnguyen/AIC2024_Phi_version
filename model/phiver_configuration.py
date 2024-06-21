from transformers import PretrainedConfig, PerceiverConfig
from .double_perceiver_config import DoublePerceiverConfig


class PhiverConfig(PretrainedConfig):
    model_type = "phi"

    def __init__(self,
                video_perceiver_config=None,
                frame_perceiver_config=None,
                text_model=None,
                num_phase=10,
                initializer_range=0.02,
                tie_word_embeddings=False,
                bos_token_id=1,
                eos_token_id=2,
                 **kwargs
                ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
            )
        
        if video_perceiver_config is None:
            self.double_perceiver = DoublePerceiverConfig(
                video_config = PerceiverConfig(
                    num_latents = 1,
                    d_latents = 1024, # change to middle layer size
                    d_model = 256, # Blip2 output is 256
                    num_blocks = 2,
                    num_self_attends_per_block = 4,
                    num_self_attention_heads = 2,
                    num_cross_attention_heads = 2,
                ).to_dict(),
                frame_config = PerceiverConfig(
                    num_latents = 20, # Zip to 20 embedings
                    d_latents = 2048, # change to size of hidden size llm
                    d_model = 1024, # Previous output
                    num_blocks = 4,
                    num_self_attends_per_block = 4,
                    num_self_attention_heads = 2,
                    num_cross_attention_heads = 2
                ).to_dict()
            )
            self.text_model = 'microsoft/phi-1_5'
        else:
            self.double_perceiver = DoublePerceiverConfig(
                video_config = video_perceiver_config,
                frame_config = frame_perceiver_config
            )
            self.text_model = text_model
            
        # Some parameters that I dont even care about, Just put them here for no bugs
        self.initializer_range = initializer_range
        self.num_phase = num_phase