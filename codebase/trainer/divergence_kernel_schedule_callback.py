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


@TrainerCallback.register("divergence_kernel_schedule_callback")
class DivergenceKernelCallback(TrainerCallback):
    """
    Callback for the scheduling of the Div Kernel weight e.g. Kullback Leibler Divergence.
    
    (i): Pre-Warmup where _divergence_kernel_scaler = 0
    (ii): Warmup where kernel scales linearly (end-start)/warmup gives %
    (iii): kernel loss added where _divergence_kernel_scaler=1
    """
    def __init__(
        self,
        serialization_dir: str,
        kernel_loss_weight_max: float = 1.0,
        warmup_start: int = 0,
        warmup_end: int = 1000
    ) -> None:
        super().__init__(serialization_dir)
        self._max_val = kernel_loss_weight_max
        self._warmup_start = warmup_start
        self._warmup_end = warmup_end
        self._warmup_steps = warmup_end - warmup_start
        self._steps = 0

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary)
        self._steps = 0

        if self._warmup_end == 0: # Const from 0
            trainer._divergence_kernel_scaler = self._max_val
        else:
            trainer._divergence_kernel_scaler = 0

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
        is_outer_step: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each batch.
        """

        if is_outer_step:
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
            trainer._divergence_kernel_scaler = val
