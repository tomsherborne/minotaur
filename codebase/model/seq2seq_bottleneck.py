from typing import Dict, Optional
import torch

from overrides import overrides
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import util, InitializerApplicator
from allennlp_models.generation.modules.seq_decoders.seq_decoder import SeqDecoder

from codebase.model import ComposedSeq2SeqKW, Bottleneck


@Model.register("seq2seq_bottleneck")
class Seq2SeqBottleneck(ComposedSeq2SeqKW):
    """
    Extends the `allennlp_models.generation.composed_seq2seq` model
    with a `Bottleneck` component between the encoder and the decoder

    ``TextFieldEmbedder` -> `Seq2SeqEncoder` -> `Bottleneck` -> `SeqDecoder``

    This is useful for variational inference modeling wherein we
    sample from some distributional space for decoding.

    We use new class `Bottleneck` to represent the intermediate
    between encoder and decoder. The default `Bottleneck`
    class is a pass through designed to mimic `PassThroughEncoder`.

    Some instances of `Bottleneck` implement some pooling so encoder ouput
        `[bsz, src_seq_len, encoder_output_dim]`
    transforms to
        `[bsz, 1, encoder_output_dim]`
    
    The `Bottleneck is also responsible for modifying the `source_padding_mask`
    if this occurs so the SeqDecoder is agnostic to the existence of the `Bottleneck`.

    We mostly subclass Seq2Seq models. But we specifically need to modify the `forward`
    logic and also fix any call to:
         `encoder.get_output_dim()` -> `bottleneck.get_output_dim()`
    """
    def __init__(
        self,
        vocab: Vocabulary,
        source_text_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        bottleneck: Bottleneck,
        decoder: SeqDecoder,
        tied_source_embedder_key: Optional[str] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
        ) -> None:
        super(Seq2SeqBottleneck, self).__init__(vocab,
                                                source_text_embedder,
                                                encoder,
                                                decoder,
                                                tied_source_embedder_key,
                                                initializer,
                                                **kwargs
        )
        self.bottleneck = bottleneck
        self.bottleneck_loss_weight = 1.0    # Controlled by callback including warmup factor   

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        outer_tokens: TextFieldTensors = None,
        outer_target_tokens: TextFieldTensors = None,
        use_outer_tokens: bool = False,
        return_encoder_states: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass on the encoder and decoder for producing the entire target sequence.

        # Parameters

        source_tokens : `TextFieldTensors`
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        outer_tokens : `TextFieldTensors` optional (default = `None`)
           The output of `TextField.as_array()` applied on the source `TextField`.
           This is an alternative input sequence for contrastive learning. We only
           use this if use_outer_tokens=True
        target_tokens : `TextFieldTensors`, optional (default = `None`)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.
        use_outer_tokens: `bool`, optional (default = `False`)
            Use outer instead of source
        return_encoder_states: `bool`, optional (default = `False`)
            Return encoder states in the output. Used for contrastive learning and error diagnosis.
            return model_outputs['bottleneck_mean'], model_outputs['bottleneck_logvar'], model_outputs['source_mask']
        # Returns

        `Dict[str, torch.Tensor]`
            The output tensors from the decoder.
        """
        if use_outer_tokens:
            state = self._encode(outer_tokens)
        else:
            state = self._encode(source_tokens)

        bottleneck_state = self._bottleneck(state, return_encoder_states=return_encoder_states)

        #### ENCODER HACK START
        viz_state = {}
        with torch.no_grad():
            viz_state["enc_max"], _ = torch.max(state["encoder_outputs"], dim=-2)
            viz_state["enc_avg"] = torch.mean(state['encoder_outputs'], dim=-2)
            viz_state["bneck_max"], _ = torch.max(bottleneck_state["encoder_outputs"], dim=-2)
            viz_state["bneck_avg"] = torch.mean(bottleneck_state['encoder_outputs'], dim=-2)
            # viz_state["bneck_var"] = bottleneck_state['bottleneck_logvar'].squeeze()

        return viz_state        
        #### ENCODER HACK END
           
        # if not self.training: # stops beam failure
        #     del bottleneck_state['bottleneck_loss']
        #
        # if use_outer_tokens and outer_target_tokens:
        #     decoded = self._decoder(bottleneck_state, outer_target_tokens)
        # else:
        #     decoded = self._decoder(bottleneck_state, target_tokens)
        #
        # # If the Bottleneck includes some loss term (e.g. KL divergence)
        # # Then we add this to the loss signal here.
        # if "loss" in decoded and "bottleneck_loss" in bottleneck_state:
        #     if bottleneck_state["bottleneck_loss"]!=None:
        #         decoded['loss'] += bottleneck_state['bottleneck_loss'] * self.bottleneck_loss_weight
        #
        # if return_encoder_states:
        #     decoded['encoder_outputs'] = bottleneck_state['encoder_outputs']
        #     decoded['source_mask'] = bottleneck_state['source_mask']
        #
        #     # These two only work with the Re-param trick. If no re-param then state['encoder_outputs']==bottleneck_state['encoder_outputs'] and bottleneck_state['bottleneck_logvar']=None
        #     decoded['bottleneck_mean'] = state['encoder_outputs']
        #     decoded['bottleneck_logvar'] = bottleneck_state['bottleneck_logvar']
        #
        # return decoded

    @overrides
    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make foward pass on the encoder.

        # Parameters

        source_tokens : `TextFieldTensors`
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        # Returns

        Dict[str, torch.Tensor]
            Map consisting of the key `source_mask` with the mask over the
            `source_tokens` text field,
            and the key `encoder_outputs` with the output tensor from
            forward pass on the encoder.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_text_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _bottleneck(self, encoder_state: Dict[str, torch.Tensor], return_encoder_states: bool = False) -> Dict[str, torch.Tensor]:

        ##
        # `Bottleneck` logic
        ##      
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        # shape: (batch_size, <bottleneck_output_dim>, encoder_output_dim)

        encoder_outputs = encoder_state["encoder_outputs"]
        source_mask = encoder_state["source_mask"]

        bneck_outputs, bneck_source_mask, bottleneck_loss, bottleneck_logvar = self.bottleneck(encoder_outputs, source_mask, return_encoder_states=return_encoder_states)

        output_dict = {"source_mask": bneck_source_mask, "encoder_outputs": bneck_outputs, "bottleneck_loss": bottleneck_loss}

        if return_encoder_states:
            output_dict['bottleneck_logvar'] = bottleneck_logvar
        else:
            del bottleneck_logvar
        torch.cuda.empty_cache()

        return output_dict
