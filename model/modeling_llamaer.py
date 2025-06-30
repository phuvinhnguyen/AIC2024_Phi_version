from .modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel, logger, DynamicCache, Cache
from transformers import PerceiverConfig, PreTrainedModel, GenerationMixin
from .configuration_llama import LlamaConfig
from .llamaer_configuration import LlamaerConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from .double_perceiver import DoublePerceiver
from typing import Union, Tuple, Optional, List, Dict
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

class EmbedPreceedLlamaModel(LlamaModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            batch_size, seq_length = input_ids.shape[:2]
            _, embed_length = inputs_embeds.shape[:2]
            seq_length = seq_length + embed_length
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify input_ids and (or) inputs_embeds")

        # batch_size, seq_length = inputs_embeds.shape[:2]

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            print('input embed is none', inputs_embeds.shape)
            print(input_ids.shape)
        else:
            inputs_embeds = torch.concat([inputs_embeds, self.embed_tokens(input_ids)], dim=1)
            print('input embed is not none', inputs_embeds.shape)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            # past_key_values_length = past_key_values.get_usable_length(seq_length)
            # device = input_ids.device if input_ids is not None else inputs_embeds.device
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            # )
            # position_ids = position_ids.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        print(position_embeddings[0].shape, position_embeddings[1].shape, position_ids.shape, hidden_states.shape)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class EmbedPreceedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = EmbedPreceedLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

class VideoLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    config_class = LlamaerConfig
    def __init__(self, config: LlamaerConfig):
        super().__init__(config)
        self.embeding_condition = nn.Embedding(num_embeddings=config.num_phase, embedding_dim=config.double_perceiver.frame_config.d_model)
        self.double_perceiver = DoublePerceiver(config.double_perceiver)
        self.llm = EmbedPreceedLlamaForCausalLM.from_pretrained(config.text_model)
        self.post_init()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        phase: torch.ShortTensor = None,
        event_embeds: Optional[torch.FloatTensor] = None,
        events_embeds_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if phase is not None:
            condition_inputs_embeds = self.embeding_condition(phase)
        else:
            condition_inputs_embeds = None
        print('inputs embeds is none:', inputs_embeds == None)
        if past_key_values is not None:
            print('past key values is not none')
            if isinstance(past_key_values, DynamicCache):
                print('length of past key values:', past_key_values.get_seq_length())
            else:
                print(past_key_values)
        else:
            print('past key values is none')

        has_past_key_values = isinstance(past_key_values, torch.Tensor) or (isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() > 0)
        if not has_past_key_values and inputs_embeds is not None:
            inputs_embeds = self.double_perceiver(inputs_embeds, condition_inputs_embeds=condition_inputs_embeds, attention_mask=inputs_embeds_mask).last_hidden_state
        else:
            inputs_embeds = None
        matching_loss = None
        if event_embeds is not None:
            event_embeds = self.double_perceiver(event_embeds, condition_inputs_embeds=condition_inputs_embeds, attention_mask=events_embeds_mask).last_hidden_state
            criteria = CrossEntropyLoss()
            matching_result = event_embeds @ inputs_embeds.transpose(1,2)
            matching_grd = torch.eye(self.double_perceiver.num_latents).argmax(dim=-1).unsqueeze(0).repeat(inputs_embeds.shape[0],1).to(matching_result.device)
            matching_loss = criteria(matching_result, matching_grd)
        print('inputs embeds is none:', inputs_embeds == None)
        causal_lm_output = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return CausalLMOutputWithPast(
            loss=causal_lm_output.loss + matching_loss if matching_loss is not None else causal_lm_output.loss,
            logits=causal_lm_output.logits,
            past_key_values=causal_lm_output.past_key_values,
            hidden_states=causal_lm_output.hidden_states,
            attentions=causal_lm_output.attentions,
        )
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, inputs_embeds_mask=None, phase=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_cache_shape()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "inputs_embeds_mask": inputs_embeds_mask,
            "phase": phase,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            }
        return model_inputs
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past 