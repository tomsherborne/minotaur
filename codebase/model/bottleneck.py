from argparse import ArgumentError
from typing import Dict, Optional, Tuple
from overrides import overrides
import torch

from allennlp.modules.encoder_base import _EncoderBase
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder
from allennlp.common import Registrable

from codebase.model.multiheadedpooling import MultiHeadedPooling


def gaussian_kl(mu, logvar):
    """
    Kullback-Liebler divergence between a Gaussian and the Prior Gaussian N(0,I).
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

def reparameterize_gaussian(mu, logvar, var_weight=1.0):
    """
    Sample a Gaussian from mu, logvar from the Encoder
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps * std * var_weight

class Bottleneck(_EncoderBase, Registrable):
    """
    A `Bottleneck` class which imposes some constraints on the output
    of an encoder as an intermediate transform between Encoder and Decoder.

    This is useful for Pooling, Variational Logic etc....

    Realistically we inherit from `_EncoderBase` as this logic
    could also be some form of `Seq2SeqEncoder` except:
    (a) the `Bottleneck` will not do any direct encoding and may be lossy
    (b) cleaner than defining some `AnotherEncoder` class
    """
    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2SeqEncoder`. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this `Seq2SeqEncoder`.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    def is_bidirectional(self) -> bool:
        raise False


@Bottleneck.register("pass_through")
class PassThroughBottleneck(Bottleneck):
    """
    Direct copy of Seq2SeqEncoder
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = PassThroughEncoder(input_dim=input_dim)

    def get_input_dim(self) -> int:
        return self.encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self.encoder.get_output_dim()
        
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Directly pass the input and output of `encoder`: `PassThroughEncoder`
        """
        return self.encoder(inputs=inputs, mask=mask), mask, None, None


@Bottleneck.register("mean")
class MeanPoolingBottleneck(Bottleneck):
    """
    Return the Mean Pooled output across all input time
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
    
    def get_input_dim(self) -> int:
        return self._input_dim
    
    def get_output_dim(self) -> int:
        "Mean pooling does not modify the dimensionality here"
        return self.get_input_dim()

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encoding mean pooling compression
        """
        with torch.no_grad():
            # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
            if mask:
                inputs = torch.mul(inputs, mask.float().unsqueeze(-1))
            
        # shape: (batch_size, 1, encoder_output_dim)
        compressed_output = torch.unsqueeze(inputs.sum(dim=-2) / mask.sum(dim=-1, keepdim=True), dim=-2, device=inputs.device)

        # shape: (batch_size, 1)
        compressed_mask = torch.ones((compressed_output.shape[0], 1), device=compressed_output.device, dtype=torch.bool)

        # No loss calculation
        bottleneck_loss = 0

        return compressed_output, compressed_mask, bottleneck_loss, None


@Bottleneck.register("max")
class MaxPoolingBottleneck(MeanPoolingBottleneck):
    """
    Return the Max Pooled output across all input time
    """
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encoding max pooling compression
        """

        # Masking values should be large negatives to avoid interfering with pooling
        
        with torch.no_grad():
            if mask:
                max_mask = mask.float()
                max_mask[~mask] = float("-1e9")
                # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
                inputs = torch.mul(inputs, max_mask.unsqueeze(-1))

        # shape: (batch_size, 1, encoder_output_dim)
        compressed_output = torch.max(inputs, dim=-2, keepdim=True)

        # shape: (batch_size, 1)
        compressed_mask = torch.ones((compressed_output.shape(0), 1), device=compressed_output.device, dtype=torch.bool)

        # No loss calculation
        bottleneck_loss = 0

        return compressed_output, compressed_mask, bottleneck_loss, None


@Bottleneck.register("multi_head")
class MultiHeadPoolingBottleneck(Bottleneck):
    """
    MultiHeadedPooling from https://arxiv.org/abs/1905.13164
    """
    def __init__(
        self,
        num_heads: int,
        model_dim: int,
        dropout: Optional[float] = 0.1,
        model_dim_out: Optional[int] = None,
        use_final_linear: Optional[bool] = True,
        use_bilinear: Optional[bool] = False,
        use_layer_norm: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.pooling = MultiHeadedPooling(
            num_heads=num_heads,
            model_dim=model_dim,
            dropout=dropout,
            model_dim_out=model_dim_out,
            use_final_linear=use_final_linear,
            use_bilinear=use_bilinear,
            use_layer_norm=use_layer_norm
        )

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # "Pool the input with Multi-Head Attention"

        # shape: (batch_size, 1, model_dim)
        # TODO(TS): Check mask usage here
        pooled = self.pooling(key=inputs, value=inputs)

        # shape: (batch_size, 1)
        pooled_mask = torch.ones((pooled.shape(0), 1), device=pooled.device, dtype=torch.bool)

        # No loss calculation
        bottleneck_loss = 0

        return pooled, pooled_mask, bottleneck_loss, None


@Bottleneck.register("variational")
class VariationalBottleneck(Bottleneck):
    """
    Variational Sampling + Reparameterisation

    \mu = Encoder Outputs
    \log\var = PooledSampleOutputs

    std = exp(0.5*\logvar)
    z = \mu + std * torch.randn_like(std)

    This model does **not** pool to a single output representation
    for the sequence. For this we will use `VariationalPoolingBottleneck`

    Note: _may_ need an extra `Linear()` output layer (https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/nn/default_architectures.py#L61)
    (Use use_final_linear=True)
    """
    def __init__(
    self,
    num_attention_heads: int,
    model_dim: int,
    dropout: Optional[float] = 0.1,
    model_dim_out: Optional[int] = None,
    use_final_linear: Optional[bool] = True,
    use_bilinear: Optional[bool] = False,
    use_layer_norm: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.logvar_pooling = MultiHeadedPooling(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            dropout=dropout,
            model_dim_out=model_dim_out,
            use_final_linear=use_final_linear,
            use_bilinear=use_bilinear,
            use_layer_norm=use_layer_norm
        )

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoding = inputs      
        var_weight = 1.0 # for now mimic hosking

        # shape: (batch_size, 1, encoder_output_dim)
        logvar = self.logvar_pooling(key=encoding, value=encoding, mask=mask)
        
        # Reparameterise over a sequence
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        gaussian_encoding = reparameterize_gaussian(mu=encoding, logvar=logvar, var_weight=var_weight)

        # if not self.training:
            # return gaussian_encoding, mask, 0, logvar
            
        # Calculate KL term from gaussian_kl. Add to returned output
        kl_losses = gaussian_kl(encoding, logvar)
        kl_losses.mul_(mask) # Multiply by mask to ignore loss on padding
        kl_loss = torch.mean(kl_losses, dim=(0,1))

        return gaussian_encoding, mask, kl_loss, logvar


@Bottleneck.register("variational_pooling")
class VariationalPoolingBottleneck(Bottleneck):
    """
    Combine `MultiHeadPoolingBottleneck` and `VariationalBottleneck`

    \mu = MultiHeadPoolingBottleneck(Encoder Outputs)
    \log\var = PooledSampleOutputs

    std = exp(0.5*\logvar)
    z = \mu + std * torch.randn_like(std)

    This model **does** pool to a single output representation
    """
    def __init__(
    self,
    num_attention_heads: int,
    model_dim: int,
    dropout: Optional[float] = 0.1,
    model_dim_out: Optional[int] = None,
    use_final_linear: Optional[bool] = True,
    use_bilinear: Optional[bool] = False,
    use_layer_norm: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.mu_pooling = MultiHeadedPooling(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            dropout=dropout,
            model_dim_out=model_dim_out,
            use_final_linear=use_final_linear,
            use_bilinear=use_bilinear,
            use_layer_norm=use_layer_norm
        )

        self.logvar_pooling = MultiHeadedPooling(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            dropout=dropout,
            model_dim_out=model_dim_out,
            use_final_linear=use_final_linear,
            use_bilinear=use_bilinear,
            use_layer_norm=use_layer_norm
        )

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # shape: (batch_size, 1, encoder_output_dim)
        encoding = self.mu_pooling(key=inputs, value=inputs, mask=mask)
        var_weight = 1.0 # for now mimic hosking

        # shape: (batch_size, 1, encoder_output_dim)
        logvar = self.logvar_pooling(key=inputs, value=inputs, mask=mask)
        
        # Reparameterise over a sequence
        # shape: (batch_size, 1, encoder_output_dim)
        gaussian_encoding = reparameterize_gaussian(mu=encoding, logvar=logvar, var_weight=var_weight)

        # Calculate KL term from gaussian_kl. Add to returned output
        kl_losses = gaussian_kl(encoding, logvar)
        kl_loss = torch.mean(kl_losses, dim=(0,1))

        # Modify the mask
        # shape: (batch_size, 1)
        mask = torch.ones((encoding.shape[0], 1), device=encoding.device, dtype=torch.bool)
        
        return gaussian_encoding, mask, kl_loss, logvar


@Bottleneck.register("wasserstein")
class WassersteinBottleneck(Bottleneck):
    """
    Wasserstein-distance based b'neck using Maximum Mean Discrepancy based minimization (Optimal Transport Theory based model)
    From: https://arxiv.org/abs/1711.01558
    Based on https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/wae_mmd/wae_mmd_model.py
    and https://github.com/tolstikhin/wae/blob/master/
    """
    def __init__(
    self,
    num_attention_heads: int,
    model_dim: int,
    mmd_kernel: Optional[str] = "imq",
    dropout: Optional[float] = 0.1,
    model_dim_out: Optional[int] = None,
    use_final_linear: Optional[bool] = True,
    use_bilinear: Optional[bool] = False,
    use_layer_norm: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.kernel_bandwidth = 1.0 # 
        self.mmd_kernel = mmd_kernel
        self.logvar_pooling = MultiHeadedPooling(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            dropout=dropout,
            model_dim_out=model_dim_out,
            use_final_linear=use_final_linear,
            use_bilinear=use_bilinear,
            use_layer_norm=use_layer_norm
        )

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoding = inputs
        var_weight = 1.0 # for now mimic hosking

        # shape: (batch_size, 1, encoder_output_dim)
        logvar = self.logvar_pooling(key=encoding, value=encoding, mask=mask)
        
        # Reparameterise over a sequence
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        gaussian_encoding = reparameterize_gaussian(mu=encoding, logvar=logvar, var_weight=var_weight)

        # if not self.training:
            # return gaussian_encoding, mask, 0

        # 1. Sample prior from randn N(0, I)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        z_prior = torch.randn_like(gaussian_encoding)

        # 2. Get Kernel between Q(z|x) and P(z)
        mmd_loss = self.sequence_imq_kernel(gaussian_encoding, z_prior)
        
        return gaussian_encoding, mask, mmd_loss, logvar

    def sequence_imq_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse MultiQuadratic Kernel based on https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py

        Standard formulation computes a (bsz, bsz) matrix and computes a kernel over this. We have an additional `seq_len`
        dimension which we treat as a batch dim here and compute the kernel per time-step before averaging at the end.

        e.g. x and y are of shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        """
        batch_size, seq_len, encoder_dim = x.size()

        # (seq_len_x, batch_size, encoder_output_dim)
        x = x.contiguous().permute(1, 0, 2)

        # (seq_len_y, batch_size, encoder_output_dim)
        y = y.contiguous().permute(1, 0, 2)

        # (seq_len_x, batch_size, 1)
        norms_x = x.pow(2).sum(dim=2, keepdim=True)

        # (seq_len_y, batch_size, 1)
        norms_y = y.pow(2).sum(dim=2, keepdim=True)

        # (seq_len_x, batch_size, batch_size)
        prods_x = torch.bmm(x, x.transpose(1, 2))
        # (seq_len_y, batch_size, batch_size)
        prods_y = torch.bmm(y, y.transpose(1, 2))

        # (seq_len_x, batch_size, batch_size)
        dists_x = norms_x + norms_x.transpose(1, 2) - 2 * prods_x
        # (seq_len_y, batch_size, batch_size)
        dists_y = norms_y + norms_y.transpose(1, 2) - 2 * prods_y

        # (seq_len_x, batch_size, batch_size)
        dot_prod = torch.bmm(x, y.transpose(1, 2))
        dist_c   = norms_x + norms_y.transpose(1, 2) - 2 * dot_prod

        res = torch.zeros((seq_len)).to(x.device)
        eps = torch.ones_like(dists_x).to(x.device) * float("1e-9")
        scales = [0.1, 0.2, 0.5, 1., 2., 5., 10.]
        for scale in scales:
            C = 2 * encoder_dim * self.kernel_bandwidth * scale

            res1 = C / (C + dists_x + eps)    # (seq_len, batch_size, batch_size)
            res1 += C / (C + dists_y + eps)   # (seq_len, batch_size, batch_size)
            
            eye = (1 - torch.eye(batch_size, device=res1.device)).unsqueeze(0) # (1, batch_size, batch_size)

            res1 = res1 * eye           # (seq_len, batch_size, batch_size)
            res1 = torch.sum(res1, dim=(-1, -2)).div_(batch_size * (batch_size - 1)) # (seq_len, )

            res2 = C / (C + dist_c + eps)     # (seq_len, batch_size, batch_size)
            res2 = torch.sum(res2, dim=(-1, -2)) * 2 / (batch_size ** 2) # (seq_len, )

            # (seq_len, )
            res += res1 - res2 

        return res.mean() # scalar

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns a matrix of shape (batch_size, batch_size) containing the pairwise kernel computation
        according to the Radial Basis Function
        """
        raise ArgumentError("RBF Kernel is currently not supported. Use IMQ Kernel.")


@Bottleneck.register("joint_individual_aggregate_wasserstein")
class JointPosteriorIndividualAggregateWassersteinBottleneck(WassersteinBottleneck):
    """
    Joint Individual and Aggregate Posterior Alignment from https://arxiv.org/abs/1812.02833 equation (7)

    L = loglike(y|z) - Beta * Individual Posterior - alpha * Aggregate Posterior

    Individual Posterior - q(z|x) is a parametric alignment over each element in q - KLDiv

    Aggregate Posterior - q(z) is a Monte Carlo sample using MMD over the whole Z space.
    """

    def __init__(
    self,
    num_attention_heads: int,
    model_dim: int,
    beta_individual: float,
    alpha_aggregate: float,  
    individual_posterior_kernel: str = "kl_div",
    mmd_kernel: Optional[str] = "imq",
    dropout: Optional[float] = 0.1,
    model_dim_out: Optional[int] = None,
    use_final_linear: Optional[bool] = True,
    use_bilinear: Optional[bool] = False,
    use_layer_norm: Optional[bool] = False,
    ) -> None:
        super(JointPosteriorIndividualAggregateWassersteinBottleneck, self).__init__(
            num_attention_heads,
            model_dim,
            mmd_kernel,
            dropout, 
            model_dim_out, 
            use_final_linear, 
            use_bilinear, 
            use_layer_norm
        )

        self.beta_individual = beta_individual
        self.alpha_aggregate = alpha_aggregate
        if individual_posterior_kernel == "kl_div":
            self.posterior_individual = self.posterior_kl_divergence
        elif individual_posterior_kernel == "l2wass":
            raise ArgumentError(f"Not implemented yet!")
        else:
            raise ArgumentError(f"Argument for individual_posterior_kernel:{individual_posterior_kernel} not recognised!")

    def posterior_kl_divergence(self, encoding_mu, encoding_logvar):
        return gaussian_kl(encoding_mu, encoding_logvar)

    def forward(
        self, 
        inputs: torch.Tensor, 
        mask: torch.BoolTensor = None, 
        return_encoder_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoding = inputs
        var_weight = 1.0 # for now mimic hosking

        # shape: (batch_size, 1, encoder_output_dim)
        logvar = self.logvar_pooling(key=encoding, value=encoding, mask=mask)
        
        # Reparameterise over a sequence
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        gaussian_encoding = reparameterize_gaussian(mu=encoding, logvar=logvar, var_weight=var_weight)

        ## Individual Posterior
        posterior_individual_loss = gaussian_kl(encoding, logvar)
        posterior_individual_loss.mul_(mask) # Multiply by mask to ignore loss on padding
        posterior_individual_loss = torch.mean(posterior_individual_loss, dim=(0,1))

        ## Aggregate Posterior
        # 1. Sample prior from randn N(0, I)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        z_prior = torch.randn_like(gaussian_encoding)

        # 2. Get Kernel between Q(z|x) and P(z)
        posterior_aggregate_loss = self.sequence_imq_kernel(gaussian_encoding, z_prior)
        
        total_loss = self.beta_individual * posterior_individual_loss + self.alpha_aggregate * posterior_aggregate_loss

        return gaussian_encoding, mask, total_loss, logvar
