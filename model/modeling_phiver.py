from .modeling_phi import PhiForCausalLM, PhiModel, PhiPreTrainedModel, logger, DynamicCache, Cache, _prepare_4d_causal_attention_mask
from transformers import PerceiverConfig, PreTrainedModel
from .configuration_phi import PhiConfig
from .phiver_configuration import PhiverConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from .double_perceiver import DoublePerceiver
from typing import Union, Tuple, Optional, List, Dict
from torch.nn import CrossEntropyLoss
from torch import nn
import torch

class EmbedPreceedPhiModel(PhiModel):
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPast]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # retrieve input_ids and inputs_embeds
            # if input_ids is not None and inputs_embeds is not None:
            #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
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

            past_key_values_length = 0

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length)

            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                inputs_embeds = torch.concat([inputs_embeds, self.embed_tokens(input_ids)], dim=1)

            inputs_embeds = self.embed_dropout(inputs_embeds)

            # Attention mask.
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

            hidden_states = inputs_embeds

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None
            # hidden_states = hidden_states.half()

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.final_layernorm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = None
            if use_cache:
                next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


class EmbedPreceedPhiForCausalLM(PhiForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = EmbedPreceedPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()


class VideoPhiForCausalLM(PhiPreTrainedModel):
    config_class = PhiverConfig

    def __init__(self, config: PhiverConfig):
        super().__init__(config)

        self.embeding_condition = nn.Embedding(num_embeddings=config.num_phase, embedding_dim=config.double_perceiver.frame_config.d_model)
        self.double_perceiver = DoublePerceiver(config.double_perceiver)
    
        self.llm = EmbedPreceedPhiForCausalLM.from_pretrained(config.text_model)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None, # (batch, seq_len)
        attention_mask: Optional[torch.Tensor] = None, # (batch, seq_len)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # (batch, frames, videos, embed_dim)
        inputs_embeds_mask: Optional[torch.FloatTensor] = None, # (batch, frames, videos)
        labels: Optional[torch.LongTensor] = None, # (batch, seq_len)
        phase: torch.ShortTensor = None, # (batch, 1)
        event_embeds: Optional[torch.FloatTensor] = None, # (batch, frames, videos, embed_dim)
        events_embeds_mask: Optional[torch.FloatTensor] = None, # (batch, frames, videos)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Get embeding of the phase and target
        if phase != None:
            condition_inputs_embeds = self.embeding_condition(phase)
        else:
            condition_inputs_embeds = None

        # Get feature vector of the inputs_embeds
        if past_key_values == None and inputs_embeds != None:
            inputs_embeds = self.double_perceiver(inputs_embeds, condition_inputs_embeds=condition_inputs_embeds, attention_mask=inputs_embeds_mask).last_hidden_state
        else:
            inputs_embeds = None

        # Get matching loss of true event and all events
        matching_loss = None
        
        if event_embeds != None:
            event_embeds = self.double_perceiver(event_embeds, condition_inputs_embeds=condition_inputs_embeds, attention_mask=events_embeds_mask).last_hidden_state
                
            # Create criteria
            criteria = CrossEntropyLoss()

            matching_result = event_embeds @ inputs_embeds.transpose(1,2)
            matching_grd = torch.eye(self.double_perceiver.num_latents).argmax(dim=-1).unsqueeze(0).repeat(inputs_embeds.shape[0],1).to(matching_result.device)

            matching_loss = criteria(matching_result, matching_grd)

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
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # if inputs_embeds is not None and past_key_values is None:
        #     model_inputs = {"inputs_embeds": inputs_embeds}
        # else:
        #     model_inputs = {"input_ids": input_ids}

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
    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
