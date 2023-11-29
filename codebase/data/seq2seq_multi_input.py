import csv
from typing import Dict, List, Optional, Iterable
import logging
import copy
from random import randint, sample

from overrides import overrides
import torch
from transformers import MBart50TokenizerFast

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Token
from allennlp.data.fields import TextField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer, SingleIdTokenIndexer

from codebase.data.seq2seq_pretrain_paired import MBARTTokenizerWrapper, LanguageFormatter, \
    VALID_MBART50_MODELS, VALID_MBART_MODELS, LANG2IDX, DEFAULT_LANGIDX

logger = logging.getLogger(__name__)

@DatasetReader.register("seq2seq_multi_input")
class PretrainedTransformerSeq2SeqMultiInputDatasetReader(DatasetReader):
    """
    Generalizes the seq2seq paired DataSet reader to allow for NL \t LF pairs and EN\tNL\tLF triples

    Expected format for each input line: Nx[<source_sequence_string>\t]<target_sequence_string>\t<source_lang>
    If we lack <source_lang> then we assume this is "en_XX".

    The output of `read` is a list of `Instance` s with the fields:
        source_tokens : `TextField` and
        outer_tokens  : `Optional[TextField]` and
        target_tokens : `Optional[TextField]` and
        source_lang   : `TextField`
    """

    def __init__(
            self,
            source_pretrained_model_name: str = None,
            source_token_namespace: str = "tokens",
            target_tokenizer: Tokenizer = None,
            target_token_indexers: Dict[str, TokenIndexer] = None,
            target_add_start_token: bool = True,
            target_add_end_token: bool = True,
            delimiter: str = "\t",
            source_max_tokens: Optional[int] = 1024,
            target_max_tokens: Optional[int] = None,
            quoting: int = csv.QUOTE_MINIMAL,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._mbart50_tokenizer = False

        if source_pretrained_model_name in VALID_MBART50_MODELS:
            logger.info(f"Creating mBART-50 based tokenizer for {source_pretrained_model_name}")
            self._source_tokenizer = MBARTTokenizerWrapper(source_pretrained_model_name, source_max_tokens)
            self._mbart50_tokenizer = True
        else:
            self._source_tokenizer = PretrainedTransformerTokenizer(
                model_name=source_pretrained_model_name, add_special_tokens=True)

        self._source_token_indexers = {
            source_token_namespace: PretrainedTransformerIndexer(model_name=source_pretrained_model_name,
                                                                 namespace=source_token_namespace
                                                                 )
        }

        # Language code validator
        self._validator = LanguageFormatter(self._source_tokenizer.tokenizer.additional_special_tokens)
       
        ####################################################
        # Target Tokenization. Options are (i) Config specified or (ii) Match Encoder 
        ####################################################
        ## (i) Target Tokenizer is specified in config
        self._target_tokenizer_independent = False # Flag to organise arguments in tti
        if target_tokenizer:
            self._target_tokenizer = target_tokenizer
            self._target_tokenizer_independent = True
        # (ii) Target Tokenizer is not specified so we match source pre-trained tokenizer
        elif self._mbart50_tokenizer:
            logger.info(f"Creating mBART-50 based tokenizer for {source_pretrained_model_name}")
            self._target_tokenizer = MBARTTokenizerWrapper(source_pretrained_model_name, source_max_tokens)
        else:
            # We assume this will work. Will throw the correct error if not
            logger.info(f"Creating generic HuggingFace tokenizer for {source_pretrained_model_name}")
            self._target_tokenizer = PretrainedTransformerTokenizer(
                model_name=source_pretrained_model_name, add_special_tokens=False)
        
        # 18 Jan 23 Modified to fix Decoder EOS / BOS for MTOP experiments
        # DecoderNet instances expect these specific symbols.
        self._start_token = Token(START_SYMBOL)
        self._end_token = Token(END_SYMBOL)

        # Start and end token logic (probably always True)
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token

        logger.info(f"Target tokenizer BOS: \"{self._start_token}\" and EOS: \"{self._end_token}\"")

        # Target indexing should probably not match source as we aren't copying the embedder.
        self._target_token_indexers = target_token_indexers # Dict of Indexers

        # TSV delimiter
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting

    def _read(self, file_path: str) -> Iterable[Instance]:

        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

        # Open data file
        with open(cached_path(file_path), "r") as data_file:
            logger.info(f"Reading instances from lines in file at: {file_path}")

            # Enumerate rows in data file
            for line_num, row in enumerate(
                    csv.reader(data_file, delimiter=self._delimiter, quoting=self.quoting)
            ):

                # FORMAT IS EITHER NL\tLF\tLOCALE or EN\tNL\tLF\tLOCALE
                if len(row) == 4:
                    source_sequence, outer_sequence, target_sequence, source_lang = row
                    outer_target_sequence = None
                # Expected format NL\tLF\tLOCALE
                elif len(row) == 3:
                    source_sequence, target_sequence, source_lang = row
                    outer_sequence = None
                    outer_target_sequence = None
                elif len(row) == 5: # MTOP Outer Loop Sequence Format: (EN, NL, LF_en, LF_nl, LOCALE)
                    source_sequence, outer_sequence, target_sequence, outer_target_sequence, source_lang = row
                else:
                    raise ConfigurationError(
                        "Invalid line format for paired data with locale: %s (line number %d)" % (row, line_num + 1)
                    )
                yield self.text_to_instance(
                    source_string=source_sequence,
                    outer_string=outer_sequence,
                    target_string=target_sequence,
                    source_lang=source_lang,
                    outer_target_string=outer_target_sequence
                    )

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    def text_to_instance(
            self, 
            source_string: str, 
            outer_string: str = None,
            target_string: str = None,
            outer_target_string: str = None, 
            source_lang: str = None,
    ) -> Instance:  # type: ignore

        # There are two ways to set source lang -- accessing the property or passing as argument.
        # This maintains flexibility as the predictors don't like named arguments
        source_lang = self._validator(source_lang)
        if outer_string: # len==4 format. 
            outer_lang = source_lang # Shift source_lang to outer_lang. 
            source_lang = self._validator("en") # If we have outer then SRC is English

        tokenized_source = self._source_tokenizer.tokenize(source_string, source_lang)
        source_lang_field = TensorField(LANG2IDX.get(source_lang, DEFAULT_LANGIDX))

        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]

        source_field = TextField(tokenized_source, self._source_token_indexers)        
        instance_dict = {"source_tokens": source_field, "source_lang": source_lang_field}

        if target_string is not None:
            if self._target_tokenizer_independent:
                tokenized_target = self._target_tokenizer.tokenize(target_string)
            else:
                tokenizer_args = (target_string, source_lang, False) if self._mbart50_tokenizer else (target_string,)
                tokenized_target = self._target_tokenizer.tokenize(*tokenizer_args)

            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]

            if self._target_add_start_token:
                tokenized_target.insert(0, copy.deepcopy(self._start_token))
            if self._target_add_end_token:
                tokenized_target.append(copy.deepcopy(self._end_token))

            target_field = TextField(tokenized_target, self._target_token_indexers)
            instance_dict = {**instance_dict, "target_tokens": target_field}

        #   Multi Language (Outer loop logic). 
        if outer_string is not None:
            tokenized_outer = self._source_tokenizer.tokenize(outer_string, outer_lang)
            outer_lang_field = TensorField(LANG2IDX.get(outer_lang, DEFAULT_LANGIDX))

            if self._source_max_tokens and len(tokenized_outer) > self._source_max_tokens:
                tokenized_outer = tokenized_outer[: self._source_max_tokens]

            outer_field = TextField(tokenized_outer, self._source_token_indexers)
            instance_dict = {**instance_dict, "outer_tokens": outer_field, "outer_lang": outer_lang_field}

        # MultiLanguage OuterLoop MTOP 
        if outer_target_string is not None:
            if self._target_tokenizer_independent:
                tokenized_otarget = self._target_tokenizer.tokenize(outer_target_string)
            else:
                tokenizer_args = (outer_target_string, "en_XX", False) if self._mbart50_tokenizer else (outer_target_string,)
                tokenized_otarget = self._target_tokenizer.tokenize(*tokenizer_args)

            if self._target_max_tokens and len(tokenized_otarget) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_otarget = tokenized_otarget[: self._target_max_tokens]

            if self._target_add_start_token:
                tokenized_otarget.insert(0, copy.deepcopy(self._start_token))
            if self._target_add_end_token:
                tokenized_otarget.append(copy.deepcopy(self._end_token))

            otarget_field = TextField(tokenized_otarget, self._target_token_indexers)
            instance_dict = {**instance_dict, "outer_target_tokens": otarget_field}
            
        return Instance(instance_dict)
