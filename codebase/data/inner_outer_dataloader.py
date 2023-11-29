from typing import Any, Dict, Iterable, Iterator, Union, Optional, List
import itertools
import math

import torch
from overrides import overrides
from copy import deepcopy
import more_itertools

from allennlp.common import util
from allennlp.data.batch import Batch
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.data_loaders.multitask_data_loader import maybe_shuffle_instances
from allennlp.data.data_loaders.multitask_scheduler import MultiTaskScheduler, HomogeneousRoundRobinScheduler, _chunked_iterator
from allennlp.data.dataset_readers.multitask import MultiTaskDatasetReader
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util

@DataLoader.register("inner_outer")
class InnerOuterDataLoader(DataLoader):
    """
    This class formats endless iterators for Inner and Outer batches.
    Inner batches are standard optimization routine runs (x, y) pairs
    Outer batches are called twice, once using "source_tokens" and again using "outer_tokens".
    This means you can calculate the divergence across batches of different input, for equivalent output.

    The class is based on `MetaLearningMultiTaskDataLoader` from the Meta-Learning project
    3.10.22
    """
    def __init__(
        self,
        reader: MultiTaskDatasetReader,
        data_path: Dict[str, str],
        scheduler: MultiTaskScheduler,
        num_instances_per_dataset: Dict[str, int],
        *,
        num_workers: Dict[str, int] = None,
        max_instances_in_memory: Dict[str, int] = None,
        start_method: Dict[str, str] = None,
        instance_queue_size: Dict[str, int] = None,
        instance_chunk_size: Dict[str, int] = None,
        shuffle: bool = True,
        cuda_device: Optional[Union[int, str, torch.device]] = None,
    ) -> None:
        # reader MultiTaskDatasetReader is 2x PretrainedTransformerSeq2SeqMultiInputDatasetReader
        self.readers = reader.readers
        self.data_paths = data_path
        self.scheduler = scheduler
        self._num_workers = num_workers or {}
        self._max_instances_in_memory = max_instances_in_memory or {}
        self._start_method = start_method or {}
        self._instance_queue_size = instance_queue_size or {}
        self._instance_chunk_size = instance_chunk_size or {}
        self._shuffle = shuffle

        self.cuda_device: Optional[torch.device] = None
        if cuda_device is not None:
            if not isinstance(cuda_device, torch.device):
                self.cuda_device = torch.device(cuda_device)
            else:
                self.cuda_device = cuda_device

        if self.readers.keys() != self.data_paths.keys():
            raise ValueError(
                f"Mismatch between readers ({self.readers.keys()}) and data paths "
                f"({self.data_paths.keys()})"
            )
 
        self._inner_key = "inner"
        self._outer_key = "outer"
        self._inner_loader = {self._inner_key: self._make_data_loader(self._inner_key)}
        self._outer_loader = {self._outer_key: self._make_data_loader(self._outer_key)}
        self._loaders = {**self._inner_loader, **self._outer_loader}
        
        self.num_instances_per_dataset = num_instances_per_dataset
        assert self.num_instances_per_dataset.keys() == self.readers.keys(), \
            "num_instances_per_dataset provided instance proportion for a dataset that is not present in readers" \
            f"num_instances_per_dataset -> {num_instances_per_dataset.keys()}" \
            f"self.readers.keys()       -> {self.readers.keys()}"
        self._instances_per_epoch = sum(num_instances_per_dataset.values())

        self._inner_iterator = {key: util.cycle_iterator_function(
            lambda l=loader: maybe_shuffle_instances(l, self._shuffle)) 
            for key, loader in self._inner_loader.items()}
        
        self._outer_iterator = {key: util.cycle_iterator_function(
            lambda l=loader: maybe_shuffle_instances(l, self._shuffle)) 
            for key, loader in self._outer_loader.items()}

    def __len__(self) -> int:
        total = 0
        batch_sizes = self.scheduler.batch_size
        total = self.num_instances_per_dataset[self._inner_key] // batch_sizes[self._inner_key]
        total += self.num_instances_per_dataset[self._outer_key] // batch_sizes[self._outer_key]
        return total
  
    def __iter__(self) -> Iterator[TensorDict]:
        raise TypeError("__iter__ is unsupported for InnerOuterDataLoader. "
                                "Use `get_inner_data/get_outer_data` instead or change to MultiTaskDataLoader")

    def get_data(self, outer=False) -> Iterator[TensorDict]:
        epoch_instances = self._get_instances_for_epoch(outer)
        
        return (
            nn_util.move_to_device(
                Batch(instances).as_tensor_dict(),
                -1 if self.cuda_device is None else self.cuda_device,
            )
            for instances in self.scheduler.batch_instances(epoch_instances)
        )
    
    def iter_instances(self) -> Iterator[Instance]:
        for loader in itertools.chain(self._inner_loader.values(), self._outer_loader.values()):
            yield from loader.iter_instances()
            
    def index_with(self, vocab: Vocabulary) -> None:
        for loader in itertools.chain(self._inner_loader.values(), self._outer_loader.values()):
            loader.index_with(vocab)

    def set_target_device(self, device: torch.device) -> None:
        self.cuda_device = device

    def _get_instances_for_epoch(self, outer: bool = False) -> Dict[str, Iterable[Instance]]:
        iterators = self._outer_iterator if outer else self._inner_iterator
        ikey = self._outer_key if outer else self._inner_key
        ninstance = self.num_instances_per_dataset[ikey]

        return {
            ikey: itertools.islice(iterators[ikey], ninstance)
        }

    def _make_data_loader(self, key: str) -> MultiProcessDataLoader:
        kwargs: Dict[str, Any] = {
            "reader": self.readers[key],
            "data_path": self.data_paths[key],
            # We don't load batches from this data loader, only instances, but we have to set
            # something for the batch size, so we set 1.
            "batch_size": 1,
        }
        if key in self._num_workers:
            kwargs["num_workers"] = self._num_workers[key]
        if key in self._max_instances_in_memory:
            kwargs["max_instances_in_memory"] = self._max_instances_in_memory[key]
        if key in self._start_method:
            kwargs["start_method"] = self._start_method[key]
        return MultiProcessDataLoader(**kwargs)
