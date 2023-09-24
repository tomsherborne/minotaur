from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("seq2seq_pretrain")
class Seq2SeqPretrainedPredictor(Predictor):
    """
    Predictor for vector to sequence models
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "...", 'vec': "..."}`.
        """
        tti_args = (json_dict["source"], None, json_dict['source_lang'])
        return self._dataset_reader.text_to_instance(*tti_args)
