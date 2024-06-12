import torch
from transformers import PreTrainedModel, PerceiverModel
from .double_perceiver_config import DoublePerceiverConfig
from typing import Dict


class DoublePerceiver(PreTrainedModel):
    config_class = DoublePerceiverConfig

    def __init__(self, config: DoublePerceiverConfig):
        '''
        parameters:
            - config: a dict of 2 perceiver configs (keys are "videos" and "frames")
        '''
        super().__init__(config)
        self.videos_perceiver = PerceiverModel(config.video_config)
        self.frames_perceiver = PerceiverModel(config.frame_config)
        self.num_latents = config.frame_config.num_latents

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, inputs: torch.Tensor, condition_inputs_embeds:torch.Tensor=None, attention_mask=None):
        '''
        parameters:
            - inputs: [batch, num_frames, num_videos, embed_dim]
        return:
            - outputs: [batch, new_num_frames, new_embed_dim]
        '''
        # Define maskings
        video_attention_mask = None
        frame_attention_mask = None

        # Get shape of attention mask
        batch, num_frames, num_videos = attention_mask.shape

        # Get video attention mask
        if attention_mask is not None:
            video_attention_mask = attention_mask.reshape([-1, num_videos])
            frame_attention_mask = attention_mask.sum(dim=-1).bool().int().to(self.frames_perceiver.device)
            frame_attention_mask = torch.concat([torch.ones(condition_inputs_embeds.shape[:-1]).to(self.frames_perceiver.device), frame_attention_mask], dim=1)

        # Get shape of inputs
        batch, num_frames, num_videos, embed_dim = inputs.shape

        # convert inputs to [batch, num_videos, embed_dim]
        inputs = inputs.reshape([-1, num_videos, embed_dim])

        # Perceive videos
        frames = self.videos_perceiver(inputs, attention_mask=video_attention_mask).last_hidden_state[:,0,:]

        # Modidy frames to [batch, num_frames, embed_dim]
        frames = frames.reshape([batch, num_frames, -1])

        # Add condition inputs if provided
        if condition_inputs_embeds is not None:
            frames = torch.concat([condition_inputs_embeds, frames], dim=1)

        return self.frames_perceiver(frames, attention_mask=frame_attention_mask)