from argparse import ArgumentError
from typing import List, Dict
import math 

import torch
from allennlp.common import Registrable

class DivergenceKernel(Registrable):
    """
    A `DivergenceKernel` class which calculates some difference between sequences
    of latent representation (arbitrarily probabilistic or non-probabilistic)
    """
    
    def compute_kernel(self) -> torch.FloatTensor:
        """
        Compute the divergence kernel from the input representations
        """
        raise NotImplementedError


@DivergenceKernel.register("pass_through")
class PassThroughDivergenceKernel(DivergenceKernel):
    """
    Do nothing kernel for testing.
    """
    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:

        null_return = torch.tensor(0.0, device=source_outputs['encoder_outputs'].device)

        return null_return
        

@DivergenceKernel.register("kullback_leibler")
class KLDivergenceKernel(DivergenceKernel):
    """
    Compute the average Kullback-Leibler divergence between two sequences of probabilistic samples.
    
    KL(p1|p2) = \ 
    0.5 * (log(|Σ_2|/|Σ_1|) - n \
        + trace(Σ_2^-1 * Σ_1) \
            + (mu2-mu1).T * (Σ_2)^-1 * (mu2-mu1))

    p1_mu.shape()     -> (batch, seq1, embed)
    p1_logvar.shape() -> (batch, 1, embed)
    p2_mu.shape()     -> (batch, seq2, embed)
    p2_logvar.shape() -> (batch, 1, embed)
    p1_mask.shape()   -> (batch, seq1)
    p2_mask.shape()   -> (batch, seq2)
    """
    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        # This is inverted because we are measuring the cost of L onto EN so EN is p2 (assym divergence artefact)
        p1_mu = target_outputs['bottleneck_mean']
        p1_logvar = target_outputs['bottleneck_logvar']
        p1_mask = target_outputs['source_mask']
        
        p2_mu = source_outputs['bottleneck_mean']
        p2_logvar = source_outputs['bottleneck_logvar']
        p2_mask = source_outputs['source_mask']

        n = p1_logvar.shape[-1]
        logp1p2 = p2_logvar.sum(dim=-1)  - p1_logvar.sum(dim=-1) - n
        tracep1p2 = torch.div(p1_logvar.squeeze().exp(), p2_logvar.squeeze().exp()).sum(dim=-1, keepdim=True)
        diff = p2_mu.unsqueeze(1).transpose(1,2) - p1_mu.unsqueeze(2).transpose(1,2)

        # logvar shape is (batch_size, 1, encoder_output_dim)
        sigma2 = torch.diag_embed(torch.exp(-p2_logvar.squeeze()))
        assert sigma2.shape == (p2_mu.shape[0],p2_mu.shape[-1], p2_mu.shape[-1])

        diffTsigma2 = torch.einsum("ijkl,ilm->ijkm", diff, sigma2)
        diffTsigma2diff = torch.einsum("ijkm,ijkm->ijk", diffTsigma2, diff)
        assert diffTsigma2diff.shape ==  (p2_mu.shape[0],p2_mu.shape[1], p1_mu.shape[1])

        cross_source_mask = p2_mask.bool().unsqueeze(1).transpose(1,2) & p1_mask.bool().unsqueeze(2).transpose(1,2)
        assert diffTsigma2diff.shape == cross_source_mask.shape

        # Add scalar to (batch, p2_seq_len, p1_seq_len) matrix
        log_trace = logp1p2 + tracep1p2
        kl_pairwise = 0.5*(log_trace.unsqueeze(-1) + diffTsigma2diff)
        
        # Check KL shape (batch, p2_seq_len, p1_seq_len)
        assert kl_pairwise.shape == cross_source_mask.shape

        # multiply KL by the cross-mask to cancel all padding<>padding interactions
        kl_pairwise.mul_(cross_source_mask)

        torch.cuda.empty_cache()

        return kl_pairwise.mean() # mean over pairs from each seq1-seq2 pairs.
        

@DivergenceKernel.register("cross_entropy")
class CrossEntropyKernel(DivergenceKernel):
    """
    Compute the average cross entropy between p1 and p2 where p2 is the true and p1 is the sample
    
    H(p1, p2) = H(p1) + KL(p1, p2)

    H(p1) = n/2 * (1 + ln(2* pi)) + 0.5 * log | Σ_2 |

    p1_mu.shape()     -> (batch, seq1, embed)
    p1_logvar.shape() -> (batch, 1, embed)
    p2_mu.shape()     -> (batch, seq2, embed)
    p2_logvar.shape() -> (batch, 1, embed)
    p1_mask.shape()   -> (batch, seq1)
    p2_mask.shape()   -> (batch, seq2)
    """
    def __init__(self):
        self.kl_div = KLDivergenceKernel()

    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        p1_logvar = target_outputs['bottleneck_logvar']
        
        cross_ent = self.kl_div.compute_kernel(source_outputs, target_outputs)

        n = p1_logvar.shape[-1]
        self_ent = n/2 * (1 + torch.log(2 * torch.tensor(math.pi))) + 0.5 * p1_logvar.sum(dim=-1).squeeze()

        return self_ent.mean() + cross_ent


@DivergenceKernel.register("cosine")
class CosineDivergenceKernel(DivergenceKernel):
    """
    Compute the mean cosine distance between samples from two batches

    Cosine(p1, p2) = 1 - sum(p1*p2) / sqrt(p1**2) * sqrt(p2**2)

    Note: this class is unstable and produces NaNs. Should not be used further w/o debugging.
    """
    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        p1 = source_outputs['encoder_outputs']
        p2 = target_outputs['encoder_outputs']
        eps = torch.tensor(1e-4).to(p1.device)
        p1_norm = torch.sqrt(p1.pow(2).sum(dim=-1))
        p2_norm = torch.sqrt(p2.pow(2).sum(dim=-1))
        p1dotp2 = torch.bmm(p1, p2.transpose(-1, -2))
        normprod = p1_norm.unsqueeze(-1) * p2_norm.unsqueeze(-2)
        cossim =  torch.div(p1dotp2, normprod + eps.expand_as(normprod))
    
        return (1-cossim).mean()


@DivergenceKernel.register("l2")
class L2DivergenceKernel(DivergenceKernel):
    """
    Compute the average L2 distance between two sequences of probabilistic samples. 
        
    L2(p1|p2) = \ 
        ||p_1 - p_2||_2^2 

        p1.shape()     -> (batch, seq1, embed)
        p2.shape()     -> (batch, seq2, embed)
    """
    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:

        p1 = source_outputs['encoder_outputs']
        p2 = target_outputs['encoder_outputs']
        p1_mask = source_outputs['source_mask']
        p2_mask = target_outputs['source_mask']

        # (batch_size, seq1, seq2, embed)
        mu_1_mu_2 = p1.unsqueeze(1).transpose(1,2) - p2.unsqueeze(2).transpose(1,2)

        # (batch_size, seq1, seq2)
        cross_source_mask = p1_mask.bool().unsqueeze(1).transpose(1,2) & p2_mask.bool().unsqueeze(2).transpose(1,2)

        # (batch_size, seq1, seq2)
        mu_1_mu_2 = mu_1_mu_2.pow(2).sum(dim=-1, keepdim=False)

        # (batch_size, seq1, seq2)
        l2 = mu_1_mu_2 * cross_source_mask

        return l2.mean()


@DivergenceKernel.register("kullback_leibler_stat")
class KLStatisticalDivergenceKernel(DivergenceKernel):
    """
    Compute the observed non-parametric KL divergence
        
    D_KL(P|Q) = sum_x P(x) * log(P(x)/Q(x))
    D_KL(P|Q) = sum_x P(x) * [log(P(x)) - log(Q(x))]
    This works by subbing X=logP, Y=logQ and re-arranging to take a final log of the output
    for numerical stability reasons. (hacky and maybe not guaranteed to be non-negative).

    p1.shape()     -> (batch, seq1, embed)
    p2.shape()     -> (batch, seq2, embed)
    """
    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:

        p1 = source_outputs['encoder_outputs']
        p2 = target_outputs['encoder_outputs']

        # (batch_size, seq1, seq2, embed)
        lp_lq = p1.unsqueeze(1).transpose(1,2) - p2.unsqueeze(2).transpose(1,2)

        # (batch_size, seq1, seq2, embed)
        p_lp_lq = (p1.exp().unsqueeze(2) * lp_lq).sum(dim=-1)

        return p_lp_lq.mean().log()


@DivergenceKernel.register("l2_wasserstein")
class L2WassersteinDivergenceKernel(DivergenceKernel):
    """
    Compute the average L2 Wasserstein divergence between two sequences of probabilistic samples. 
        
    W_l2(p1|p2) = \ 
        ||mu_1 - mu_2||_2^2 + trace(sigma1 + sigma2 - 2(sigma_1**0.5 * sigma_2 * sigma_1**0.5)**0.5)

    p1_mu.shape()     -> (batch, seq1, embed)
    p1_logvar.shape() -> (batch, 1, embed)
    p2_mu.shape()     -> (batch, seq2, embed)
    p2_logvar.shape() -> (batch, 1, embed)
    p1_mask.shape()   -> (batch, seq1)
    p2_mask.shape()   -> (batch, seq2)
    """
    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:

        p1_mu = source_outputs['bottleneck_mean']
        p1_logvar = source_outputs['bottleneck_logvar']
        p2_mu = target_outputs['bottleneck_mean']
        p2_logvar = target_outputs['bottleneck_logvar']
        p1_mask = source_outputs['source_mask']
        p2_mask = target_outputs['source_mask']

        # (batch_size, seq1, seq2, embed)
        mu_1_mu_2 = p1_mu.unsqueeze(1).transpose(1,2) - p2_mu.unsqueeze(2).transpose(1,2)

        # (batch_size, seq1, seq2)
        cross_source_mask = p1_mask.bool().unsqueeze(1).transpose(1,2) & p2_mask.bool().unsqueeze(2).transpose(1,2)

        # (batch_size, seq1, seq2)
        mu_1_mu_2 = mu_1_mu_2.pow(2).sum(dim=-1, keepdim=False)

        p1_var = p1_logvar.exp().squeeze()
        p2_var = p2_logvar.exp().squeeze()
        p1_root = torch.sqrt(p1_var)
        var_sum = (p1_var + p2_var - 2 * torch.sqrt(p1_root * p2_var * p1_root)).sum(dim=-1)

        # (batch_size, seq1, seq2)
        l2_trace = (mu_1_mu_2 + var_sum[:, None, None]) * cross_source_mask
        
        return l2_trace.mean()


@DivergenceKernel.register("mmd_imq")
class MaximumMeanDiscrepancyIMQKernel(DivergenceKernel):
    """
    Compute the MMD Distance (https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
    using the IMQ Kernel (follows the WAE paper)
    
    This is the same as the MMD using IMQ kernel but instead of the feature-mean for each time step
    using each t-th sample in the batch, we learn feature-means for each sample and average across batch. 
    This is closer to the behaviour we want for cross-lingual transfer where we learn a feature-mean for
    each sample (across the T time-steps in the sample) to approximate the local space of each language. 
    Minimising the distance between the local space between samples is the desired xlingual behaviour.
    
    e.g. x and y are of shape: (batch_size, max_input_sequence_length, encoder_output_dim)
    """
    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:

        p1 = source_outputs['encoder_outputs']
        p2 = target_outputs['encoder_outputs']
        p1_mask = source_outputs['source_mask']
        p2_mask = target_outputs['source_mask']
    
        batch_size, t_x, encoder_dim = p1.size()
        _, t_y, _ = p2.size()

        p1 = p1 * p1_mask.unsqueeze(-1)
        p2 = p2 * p2_mask.unsqueeze(-1)

        # (batch_size, seq_len_x, 1)
        norms_x = p1.pow(2).sum(dim=2, keepdim=True)
        # (batch_size, seq_len_y, 1)
        norms_y = p2.pow(2).sum(dim=2, keepdim=True)

        # (batch_size, seq_len_x, seq_len_x)
        prods_x = torch.bmm(p1, p1.transpose(1, 2))
        # (batch_size, seq_len_y, seq_len_y)
        prods_y = torch.bmm(p2, p2.transpose(1, 2))

        # (batch_size, seq_len_x, seq_len_x)
        dists_x = norms_x + norms_x.transpose(1, 2) - 2 * prods_x
        # (batch_size, seq_len_y, seq_len_y)
        dists_y = norms_y + norms_y.transpose(1, 2) - 2 * prods_y

        # (batch_size, seq_len_x, seq_len_y)
        dot_prod = torch.bmm(p1, p2.transpose(1, 2))

        # (batch_size, seq_len_x, seq_len_y)
        dist_c   = norms_x + norms_y.transpose(1, 2) - 2 * dot_prod

        comp_x = torch.zeros((batch_size)).to(p1.device)
        comp_y = torch.zeros((batch_size)).to(p1.device)
        comp_xy = torch.zeros((batch_size)).to(p1.device)
        
        scales = [0.1, 0.2, 0.5, 1., 2., 5., 10.]
        for scale in scales:
            C = 2 * encoder_dim * 1 * scale

            res_x = C / (C + dists_x)    # (batch_size, seq_len_x, seq_len_x)
            res_y = C / (C + dists_y)   # (batch_size, seq_len_y, seq_len_y)
            
            eye_x = (1 - torch.eye(t_x, device=res_x.device)).unsqueeze(0) # (1, seq_len_x, seq_len_x)
            eye_y = (1 - torch.eye(t_y, device=res_y.device)).unsqueeze(0) # (1, seq_len_y, seq_len_y)
            res_x *= eye_x  # (batch_size, seq_len_x, seq_len_x)
            res_y *= eye_y  # (batch_size, seq_len_y, seq_len_y)

            res_xy = C / (C + dist_c) # (batch_size, seq_len_x, seq_len_y)

            comp_x += res_x.sum(dim=(-1, -2))  # (batch_size, )
            comp_y += res_y.sum(dim=(-1, -2))  # (batch_size, )
            comp_xy = res_xy.sum(dim=(-1, -2)) # (batch_size, )

        res = (1/(t_x * (t_x - 1))) * comp_x + (1/(t_y * (t_y - 1))) * comp_y - (2/(t_x * t_y)) * comp_xy

        return res.mean() # scalar


@DivergenceKernel.register("joint_posterior")
class JointPosteriorIndividualAggregateWassersteinBottleneck(DivergenceKernel):
    """
    Joint Individual and Aggregate Posterior Alignment from https://arxiv.org/abs/1812.02833 equation (7)

    L = loglike(y|z) - Beta * Individual Posterior - alpha * Aggregate Posterior

    Individual Posterior - q(z|x) is a parametric alignment over each element in q1 and q2

    Aggregate Posterior - q(z) is a Monte Carlo sample using MMD over the whole Z space.
    """
    def __init__(self, individual_posterior_kernel: str = "kl_div"):
        if individual_posterior_kernel == "kl_div":
            self.posterior_individual = KLDivergenceKernel()
        elif individual_posterior_kernel == "l2wass":
            self.posterior_individual = L2WassersteinDivergenceKernel()
        else:
            raise ArgumentError(f"Argument for individual_posterior_kernel:{individual_posterior_kernel} not recognised!")
        self.posterior_aggregate = MaximumMeanDiscrepancyIMQKernel()


    def compute_kernel(
        self,
        source_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        individual_loss = self.posterior_individual.compute_kernel(source_outputs, target_outputs)
        aggregate_loss = self.posterior_aggregate.compute_kernel(source_outputs, target_outputs)

        return individual_loss, aggregate_loss
