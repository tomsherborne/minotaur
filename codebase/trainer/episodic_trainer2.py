import datetime
import itertools
import logging
import math
from pyexpat import model
import re
from typing import Optional, Union, List, Dict, Tuple, Any, Type, Iterable, Iterator, TypeVar
from collections import defaultdict
from copy import deepcopy
import torch
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import util as common_util, Tqdm, Lazy
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer, TrainerCheckpoint
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer, DEFAULT_CALLBACKS
from allennlp.training.callbacks import TrainerCallback
from allennlp.training import util as training_util 

from codebase.trainer import DivergenceKernel, KLDivergenceKernel

logger = logging.getLogger(__name__)

A = TypeVar("A")

def lazy_variable_groups_of(iterable: Iterable[A], group_sizes: Union[Iterable[int], int]) -> Iterator[List[A]]:
    """
    Modified from common_util.lazy_groups_of for groups of variable length.
    This is to permit inner_steps to be randomly sampled from some (min, max) 
    during an epoch to imitate a MultiTaskSampler behaviour for multitask learning.
    """
    iterator = iter(iterable)

    if type(group_sizes) is int: # If group_sizes is a constant then behave same as lazy_groups_of
        group_sizes = itertools.cycle([group_sizes])
    else:
        group_sizes = iter(group_sizes)

    while True:
        s = list(itertools.islice(iterator, next(group_sizes)))
        if len(s) > 0:
            yield s
        else:
            break


@Trainer.register("episodic_trainer_v2", constructor="from_partial_objects")
class EpisodicTrainerV2(GradientDescentTrainer):
    """
    Episodic Trainer modeled on the MetaTrainer from xgr/metaparse project.
    Contrasting to the meta-learning approach, we have **one** optimizer.

    The divergence kernel can be the Kullback-Liebler, Wasserstein MMD. 

    Note that this requires divergence across **sequences**. 

    In V2: we use Registrable divergence Kernels 
    
    Inner task: The trainer runs K steps of standard SGD on the source language
    Outer task: Similar to Inner for 2 languages + divergence loss between encoded representations i.e.
                outer_loss = cross_ent_lang_1 + prior_divergence_lang_1 + \
                             cross_ent_lang_2 + prior_divergence_lang_2 + \
                             divergence_kernel_lang_1_lang_2
    """
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 data_loader: DataLoader,
                 inner_steps: int,
                 use_outer_loop: bool = True,
                 use_divergence_kernel: bool = True,
                 divergence_kernel: Optional[DivergenceKernel] = None,
                 divergence_kernel_scaler: Union[float, Tuple[float]] = 1.0,
                 patience: Optional[int] = None,
                 validation_metric: Union[str, List[str]] = "-loss",
                 validation_data_loader: DataLoader = None,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 checkpointer: Checkpointer = None,
                 cuda_device: Optional[Union[int, torch.device]] = None,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 momentum_scheduler: Optional[MomentumScheduler] = None,
                 moving_average: Optional[MovingAverage] = None,
                 callbacks: List[TrainerCallback] = None,
                 distributed: bool = False,
                 local_rank: int = 0,
                 world_size: int = 1,
                 num_gradient_accumulation_steps: int = 1,
                 use_amp: bool = False,
                 enable_default_callbacks: bool = True,
                 run_confidence_checks: bool = False,
                 **kwargs) -> None:
        super().__init__(model, optimizer, data_loader, patience, validation_metric, validation_data_loader, num_epochs,
                         serialization_dir, checkpointer, cuda_device, grad_norm, grad_clipping,
                         learning_rate_scheduler, momentum_scheduler, moving_average, callbacks, distributed,
                         local_rank, world_size, num_gradient_accumulation_steps, use_amp, enable_default_callbacks,
                         run_confidence_checks, **kwargs)

        # Episodic Learning Parameters
        self._inner_steps = inner_steps
        self._total_outer_steps_completed = 0
        self._use_outer_loop = use_outer_loop
        if (not self._use_outer_loop) or (not divergence_kernel):
            use_divergence_kernel = False # Cannot have div kernel w/o outer loop
        self._use_divergence_kernel = use_divergence_kernel
        self._divergence_kernel = divergence_kernel
        self._divergence_kernel_scaler = divergence_kernel_scaler

        # assert type(self.data_loader) is MetaLearningMultiTaskDataLoader
        self.tqdm_loader = None
        self.epoch_running_train_loss = 0
        self.epoch_running_train_reg_loss = 0
        self.dkernel_logfile = serialization_dir + "/dkernel.log"
        self.dkernel_headers = ["step", "loss", "scale"]
        self.dkernel_log_setup()

    def dkernel_log_setup(self) -> None:
        logger.info(f"Saving kernel data to {self.dkernel_logfile}")
        with open(self.dkernel_logfile, "w") as fh:
            fh.write("\t".join(self.dkernel_headers) + "\n")
        return
    
    def dkernel_log(self, data_dict: Dict[str, float]) -> None:
        """
        Hack to save the divergence kernel loss values
        """
        for head in self.dkernel_headers:
            if head not in data_dict:
                print("Data invalid, skipping!")
                return
        with open(self.dkernel_logfile, "a") as fh:
            for header in self.dkernel_headers:
                fh.write(str(data_dict[header]) + "\t")
            fh.write("\n")
        return
    
    def rescale_gradients(self, optimizer: torch.optim.Optimizer) -> Union[float, torch.Tensor]:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.

        Returns the norm of the gradients.
        """
        parameters_to_clip = [p for p in self.model.parameters() if p.grad is not None]
        if self._grad_norm:
            if self._scaler is not None:
                # Need to first unscale gradients in order to clip as usual.
                self._scaler.unscale_(optimizer)
            return clip_grad_norm_(parameters_to_clip, self._grad_norm)
        else:
            return torch.norm(
                torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip])
            )

    def outer_loop(self,
                   inner_loop_batch_groups: Iterable[Iterable[TensorDict]],
                   outer_loop_batch_group: Iterable[TensorDict],
                   epoch: int,
                   ) -> Tuple[float, float]:
        

        for batch_group in inner_loop_batch_groups:
            il_train_loss, il_train_reg_loss = self.inner_loop(batch_group, epoch)

        if not self._use_outer_loop:
            torch.cuda.empty_cache()
            return 0, 0

        batch_loss = 0.0
        batch_reg_loss = None if self.model.get_regularization_penalty() is None else 0.0
        batch_group_outputs = []

        for batch in outer_loop_batch_group:
            with amp.autocast(self._use_amp):
                ########################################################
                ## OUTER LOOP BATCH GROUP PROCESSING HERE.
                ########################################################

                source_outputs = self.batch_outputs(batch, for_training=True, return_encoder_states=True)
                target_outputs = self.batch_outputs(batch, for_training=True, return_encoder_states=True, use_outer_tokens=True)

                # SOURCE LOSS ###
                batch_group_outputs.append(source_outputs)
                source_loss = source_outputs["loss"]
                source_reg_loss = source_outputs.get("reg_loss")
                if torch.isnan(source_loss):
                    continue
                    # raise ValueError("nan loss encountered")
                source_loss = source_loss / len(batch_group)

                if source_reg_loss is not None:
                    source_reg_loss = source_reg_loss / len(batch_group)
                    batch_reg_loss = source_reg_loss.item()
                    self.epoch_running_train_reg_loss += batch_reg_loss  # type: ignore

                # TARGET LOSS ###
                batch_group_outputs.append(target_outputs)
                target_loss = target_outputs["loss"]
                target_reg_loss = target_outputs.get("reg_loss")
                if torch.isnan(target_loss):
                    continue
                    # raise ValueError("nan loss encountered")
                target_loss = target_loss / len(batch_group)

                if target_reg_loss is not None:
                    target_reg_loss = target_reg_loss / len(batch_group)
                    batch_reg_loss = target_reg_loss.item()
                    self.epoch_running_train_reg_loss += batch_reg_loss  # type: ignore

                outer_loss = (source_loss + target_loss) * 0.5

                if self._use_divergence_kernel:
                    divergence_kernel_output = self._divergence_kernel.compute_kernel(source_outputs, target_outputs)
                    # Hacky log
                    self.dkernel_log({"step": self._total_batches_completed, "loss": divergence_kernel_output, "scale": self._divergence_kernel_scaler})

                    # Some kernels return Tuples. This is used for JointPosteriorIndividualAggregateWassersteinBottleneck
                    if isinstance(divergence_kernel_output, tuple):
                        if not isinstance(self._divergence_kernel_scaler, Iterable): # Divergence_kernel_scaler = (Beta_indiv, Alpha_agg)
                            self._divergence_kernel_scaler = defaultdict(lambda: self._divergence_kernel_scaler)

                        divergence_kernel_loss = divergence_kernel_output[0] * self._divergence_kernel_scaler[0] + \
                            divergence_kernel_output[1] * self._divergence_kernel_scaler[1]
                        divergence_kernel_loss *= (1 / len(batch_group))
                    else: # Output is a scalar
                        divergence_kernel_loss = divergence_kernel_output * (self._divergence_kernel_scaler / len(batch_group))

                    
                    outer_loss += divergence_kernel_loss
                
                batch_loss += outer_loss.item()

                torch.cuda.empty_cache()

            if self._scaler is not None:
                self._scaler.scale(outer_loss).backward()
            else:
                outer_loss.backward()

        if len(batch_group_outputs) <= 0:
            return 0, 0

        self.epoch_running_train_loss += batch_loss

        batch_grad_norm = self.rescale_gradients(self.optimizer)

        if self._learning_rate_scheduler:
            self._learning_rate_scheduler.step_batch(self._total_batches_completed + 1)
        if self._momentum_scheduler:
            self._momentum_scheduler.step_batch(self._total_batches_completed + 1)

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        # Update moving averages
        if self._moving_average is not None:
            self._moving_average.apply(self._total_batches_completed + 1)

        self._batches_in_epoch_completed += 1
        self._total_batches_completed += 1

        # Reset num inner steps
        self._total_outer_steps_completed += 1

        # Update the description with the latest metrics
        metrics = training_util.get_metrics(
            self.model,
            self.epoch_running_train_loss,
            self.epoch_running_train_reg_loss,
            batch_loss,
            batch_reg_loss,
            self._batches_in_epoch_completed,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )

        # Inject Divergence Kernel Loss into metrics callback
        metrics['divergence_kernel_loss'] = divergence_kernel_loss

        for callback in self._callbacks:
            callback.on_batch(
                self,
                batch_group,
                batch_group_outputs,
                metrics,
                epoch,
                self._batches_in_epoch_completed,
                is_training=True,
                is_primary=self._primary,
                batch_grad_norm=batch_grad_norm,
                is_outer_step=True
        )

        if self._primary:
            # Updating tqdm only for the primary as the trainers wouldn't have one
            description = training_util.description_from_metrics(metrics)
            self.tqdm_loader.set_description(description, refresh=False)
            self.tqdm_loader.update(1)

            if self._checkpointer is not None:
                self._checkpointer.maybe_save_checkpoint(
                    self, self._epochs_completed, self._batches_in_epoch_completed
                )

        del outer_loss, source_loss, target_loss, batch_group, batch_group_outputs
        torch.cuda.empty_cache()

        return 0, 0

    def inner_loop(self, batch_group: Iterable[TensorDict], epoch: int) -> Tuple[float, float]:
        self.optimizer.zero_grad()
        batch_loss = 0.0
        batch_reg_loss = None if self.model.get_regularization_penalty() is None else 0.0
        batch_group_outputs = []

        for batch in batch_group:
            with amp.autocast(self._use_amp):
                batch_outputs = self.batch_outputs(batch, for_training=True)
                batch_group_outputs.append(batch_outputs)
                loss = batch_outputs["loss"]
                reg_loss = batch_outputs.get("reg_loss")
                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")
                loss = loss / len(batch_group)

                batch_loss += loss.item()
                if reg_loss is not None:
                    reg_loss = reg_loss / len(batch_group)
                    batch_reg_loss = reg_loss.item()
                    self.epoch_running_train_reg_loss += batch_reg_loss  # type: ignore

            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        if len(batch_group_outputs) <= 0:
            return 0, 0

        self.epoch_running_train_loss += batch_loss

        batch_grad_norm = self.rescale_gradients(self.optimizer)

        if self._learning_rate_scheduler:
            self._learning_rate_scheduler.step_batch(self._total_batches_completed + 1)
        if self._momentum_scheduler:
            self._momentum_scheduler.step_batch(self._total_batches_completed + 1)

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        # Update moving averages
        if self._moving_average is not None:
            self._moving_average.apply(self._total_batches_completed + 1)

        self._batches_in_epoch_completed += 1
        self._total_batches_completed += 1

        # Update the description with the latest metrics
        metrics = training_util.get_metrics(
            self.model,
            self.epoch_running_train_loss,
            self.epoch_running_train_reg_loss,
            batch_loss,
            batch_reg_loss,
            self._batches_in_epoch_completed,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )

        for callback in self._callbacks:
            callback.on_batch(
                self,
                batch_group,
                batch_group_outputs,
                metrics,
                epoch,
                self._batches_in_epoch_completed,
                is_training=True,
                is_primary=self._primary,
                batch_grad_norm=batch_grad_norm,
            )

        if self._primary:
            # Updating tqdm only for the primary as the trainers wouldn't have one
            description = training_util.description_from_metrics(metrics)
            self.tqdm_loader.set_description(description, refresh=False)
            self.tqdm_loader.update(1)

            if self._checkpointer is not None:
                self._checkpointer.maybe_save_checkpoint(
                    self, self._epochs_completed, self._batches_in_epoch_completed
                )
        return self.epoch_running_train_loss, self.epoch_running_train_reg_loss

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []
        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage: {common_util.format_size(memory)}")
        gpu_memory_usage = []
        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage: {common_util.format_size(memory)}")

        regularization_penalty = self.model.get_regularization_penalty()

        done_early = False
        self.epoch_running_train_loss = 0.0
        self.epoch_running_train_reg_loss = None if regularization_penalty is None else 0.0

        logger.info(f"EpisodicTrainer V2 with outer->{self._use_outer_loop} div_kernel->{self._divergence_kernel} *************")

        # Set the model to "train" mode.
        self._pytorch_model.train()

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        # Iterator[TensorDict]
        batch_generator = self.data_loader.get_data(outer=False)

        # Gradient accumulation batches -> Iterable[Iterator[TensorDict]]
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        # Inner-loop groupings -> Iterable[Iterable[Iterator[TensorDict]]]
        # If inner_steps is a sequence we return variable groups (depends on inner_random_sample)
        inner_loop_generator = lazy_variable_groups_of(batch_group_generator, self._inner_steps)
        
        # Get Target Data from dataloader
        # Iterable[TensorDict]
        outer_batch_generator = self.data_loader.get_data(outer=True)

        # One effective batch of target data may still be too large so we accumulate grads over smaller batches
        # Iterable[Iterable[TensorDict]]]
        outer_batch_group_generator = common_util.lazy_groups_of(
            outer_batch_generator, self._num_gradient_accumulation_steps
        )

        if self._primary:
            # TODO: Check if num_training_batches is actually the length of the data batches
            self.tqdm_loader = Tqdm.tqdm(
                inner_loop_generator, total=num_training_batches
            ) # iterable argument may be redundant here
        else:
            self.tqdm_loader = inner_loop_generator

        for i, inner_group in enumerate(inner_loop_generator):
            if self._epochs_completed < self._start_after_epochs_completed or (
                self._epochs_completed == self._start_after_epochs_completed
                and self._batches_in_epoch_completed < self._start_after_batches_in_epoch_completed
            ):
                self._batches_in_epoch_completed += self._inner_steps
                self._total_batches_completed += self._inner_steps
                continue

            if self._use_outer_loop:
                # inner_group is the one Kgroup for one pass of K inner-steps. Each of K steps is accum forward passes and 1 backward pass
                # We call this directly from the lazy iterator as we only need 1
                outer_batch_group = next(outer_batch_group_generator)
            else:
                outer_batch_group = None

            # Run an outer-loop on the current outer-loop-group
            _, _ = self.outer_loop(inner_loop_batch_groups=inner_group,
                                   outer_loop_batch_group=outer_batch_group,
                                   epoch=epoch
                                   )

        del outer_batch_group_generator, inner_loop_generator
        torch.cuda.empty_cache()

        logger.info(f"\nEpoch {epoch} training over")

        metrics = training_util.get_metrics(
            self.model,
            self.epoch_running_train_loss,
            self.epoch_running_train_reg_loss,
            batch_loss=None,
            batch_reg_loss=None,
            num_batches=self._batches_in_epoch_completed,
            reset=True,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )
        self.tqdm_loader.close()  # Cleanup TQDM

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (1024 * 1024)
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (1024 * 1024)

        return metrics

    def batch_outputs(
        self, 
        batch: TensorDict, 
        for_training: bool, 
        return_encoder_states: bool = False, 
        use_outer_tokens: bool = False
        ) -> Dict[str, torch.Tensor]:
        """
        Does a forward pass on the given batch and returns the output dictionary that the model
        returns, after adding any specified regularization penalty to the loss (if training).
        """
        output_dict = self._pytorch_model(**batch, return_encoder_states=return_encoder_states, use_outer_tokens=use_outer_tokens)

        if for_training:
            try:
                assert "loss" in output_dict
                regularization_penalty = self.model.get_regularization_penalty()

                if regularization_penalty is not None:
                    output_dict["reg_loss"] = regularization_penalty
                    output_dict["loss"] += regularization_penalty

            except AssertionError:
                if for_training:
                    raise RuntimeError(
                        "The model you are trying to optimize does not contain a"
                        " 'loss' key in the output of model.forward(inputs)."
                    )

        return output_dict

    @classmethod
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: DataLoader = None,
        local_rank: int = 0,
        patience: int = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: float = None,
        grad_clipping: float = None,
        distributed: bool = False,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        no_grad: List[str] = None,
        divergence_kernel: Lazy[DivergenceKernel] = None,
        optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
        learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
        momentum_scheduler: Lazy[MomentumScheduler] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = Lazy(Checkpointer),
        callbacks: List[Lazy[TrainerCallback]] = None,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        **kwargs,
    ) -> Trainer:
       
        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)
        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)

        common_util.log_frozen_and_tunable_parameter_names(model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        moving_average_ = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        learning_rate_scheduler_ = (
            None
            if learning_rate_scheduler is None
            else learning_rate_scheduler.construct(
                optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
            )
        )
        momentum_scheduler_ = (
            None
            if momentum_scheduler is None
            else momentum_scheduler.construct(optimizer=optimizer_)
        )
        checkpointer_ = checkpointer.construct(serialization_dir=serialization_dir)

        callbacks_: List[TrainerCallback] = []
        for callback_ in callbacks or []:
            callbacks_.append(callback_.construct(serialization_dir=serialization_dir))


        divergence_kernel_ = (
            None
            if divergence_kernel is None
            else divergence_kernel.construct()
        )

        return cls(
            model,
            optimizer_,
            data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            callbacks=callbacks_,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            enable_default_callbacks=enable_default_callbacks,
            run_confidence_checks=run_confidence_checks,
            divergence_kernel=divergence_kernel_,
            **kwargs,
        )