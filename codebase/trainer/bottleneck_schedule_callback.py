from curses.ascii import LF
import datetime
import logging
import math
import os
import re
import time
import traceback
import shutil
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch.optim.lr_scheduler

from allennlp.training.callbacks import TrainerCallback
from allennlp.data import DataLoader, TensorDict
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

logger = logging.getLogger(__name__)


@TrainerCallback.register("bottleneck_loss_callback")
class BottleneckLossCallback(TrainerCallback):
    """
    Callback for the scheduling of the Bottleneck loss e.g. Kullback Leibler Divergence.
    The Bottleneck loss is calculated in the instance of `Bottleneck` and applied in the
    `Seq2SeqBottleneck`  model using the `Seq2SeqBottleneck.bottleneck_loss_weight` 
    variable. This callback schedules the variable over training in three phases:
    (i): Pre-Warmup where bottleneck_loss_weight = 0
    (ii): Warmup where KL scales linearly (end-start)/warmup gives %
    (iii): KL loss added where bottleneck_loss_weight=1
    """
    def __init__(
        self,
        serialization_dir: str,
        bottleneck_loss_weight_max: float = 1.0,
        warmup_start: int = 0,
        warmup_end: int = 1000
    ) -> None:
        super().__init__(serialization_dir)
        self._max_val = bottleneck_loss_weight_max
        self._warmup_start = warmup_start
        self._warmup_end = warmup_end
        self._warmup_steps = warmup_end - warmup_start
        self._steps = 0

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary)
        self._steps = 0
        trainer.model.bottleneck_loss_weight = 0

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each batch.
        """
        self._steps += 1

        # Before the warmup the KL weight is 0
        if self._steps < self._warmup_start:
            val = 0.
        # Mid warmup uses a linear increasing scale up to 1.0
        elif self._steps < self._warmup_end:
            val = float((self._steps - self._warmup_start) / self._warmup_steps)
        # Post warmup uses the KL loss `as-is`
        else:    
            val = self._max_val
        trainer.model.bottleneck_loss_weight = val
