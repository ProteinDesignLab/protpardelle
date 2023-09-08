"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Top-level model definitions.
Typically these are initialized with config rather than arguments.
"""
import argparse
from functools import partial
import os
from typing import Callable, List, Optional

from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from core import protein_mpnn
from core import residue_constants
from core import utils
import diffusion
import evaluation
import modules


class MiniMPNN(nn.Module):
    """Wrapper for ProteinMPNN network to predict sequence from structure."""

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        self.model_config = cfg = config.model.mpnn_model
        self.n_tokens = config.data.n_aatype_tokens
        self.seq_emb_dim = cfg.n_channel
        time_cond_dim = cfg.n_channel * cfg.noise_cond_mult

        self.noise_block = modules.NoiseConditioningBlock(cfg.n_channel, time_cond_dim)
        self.token_embedding = nn.Linear(self.n_tokens, self.seq_emb_dim)
        self.mpnn_net = modules.NoiseConditionalProteinMPNN(
            n_channel=cfg.n_channel,
            n_layers=cfg.n_layers,
            n_neighbors=cfg.n_neighbors,
            time_cond_dim=time_cond_dim,
            vocab_size=config.data.n_aatype_tokens,
            input_S_is_embeddings=True,
        )
        self.proj_out = nn.Linear(cfg.n_channel, self.n_tokens)

    def forward(
        self,
        denoised_coords: TensorType["b n a x", float],
        coords_noise_level: TensorType["b", float],
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
        seq_self_cond: Optional[TensorType["b n t", float]] = None,  # logprobs
        return_embeddings: bool = False,
    ):
        coords_noise_level_scaled = 0.25 * torch.log(coords_noise_level)
        noise_cond = self.noise_block(coords_noise_level_scaled)

        b, n, _, _ = denoised_coords.shape
        if seq_self_cond is None or not self.model_config.use_self_conditioning:
            seq_emb_in = torch.zeros(b, n, self.seq_emb_dim).to(denoised_coords)
        else:
            seq_emb_in = self.token_embedding(seq_self_cond.exp())

        node_embs, encoder_embs = self.mpnn_net(
            denoised_coords, seq_emb_in, seq_mask, residue_index, noise_cond
        )

        logits = self.proj_out(node_embs)
        pred_logprobs = F.log_softmax(logits, -1)

        if return_embeddings:
            return pred_logprobs, node_embs, encoder_embs
        return pred_logprobs


class CoordinateDenoiser(nn.Module):
    """Wrapper for U-ViT module to denoise structure coordinates."""

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config

        # Configuration
        self.sigma_data = config.data.sigma_data
        m_cfg = config.model.struct_model
        nc = m_cfg.n_channel
        bb_atoms = ["N", "CA", "C", "O"]
        n_atoms = config.model.struct_model.n_atoms
        self.use_conv = len(m_cfg.uvit.n_filt_per_layer) > 0
        if self.use_conv and n_atoms == 37:
            n_atoms += 1  # make it an even number
        self.n_atoms = n_atoms
        self.bb_idxs = [residue_constants.atom_order[a] for a in bb_atoms]
        n_xyz = 9 if config.model.crop_conditional else 6
        nc_in = n_xyz * n_atoms  # xyz + selfcond xyz + maybe cropcond xyz

        # Neural networks
        n_noise_channel = nc * m_cfg.noise_cond_mult
        self.net = modules.TimeCondUViT(
            seq_len=config.data.fixed_size,
            patch_size=m_cfg.uvit.patch_size,
            dim=nc,
            depth=m_cfg.uvit.n_layers,
            n_filt_per_layer=m_cfg.uvit.n_filt_per_layer,
            heads=m_cfg.uvit.n_heads,
            dim_head=m_cfg.uvit.dim_head,
            conv_skip_connection=m_cfg.uvit.conv_skip_connection,
            n_atoms=n_atoms,
            channels_per_atom=n_xyz,
            time_cond_dim=n_noise_channel,
            position_embedding_type=m_cfg.uvit.position_embedding_type,
        )
        self.noise_block = modules.NoiseConditioningBlock(nc, n_noise_channel)

    def forward(
        self,
        noisy_coords: TensorType["b n a x", float],
        noise_level: TensorType["b", float],
        seq_mask: TensorType["b n", float],
        residue_index: Optional[TensorType["b n", int]] = None,
        struct_self_cond: Optional[TensorType["b n a x", float]] = None,
        struct_crop_cond: Optional[TensorType["b n a x", float]] = None,
    ):
        # Prep inputs and time conditioning
        actual_var_data = self.sigma_data**2
        var_noisy_coords = noise_level**2 + actual_var_data
        emb = noisy_coords / utils.expand(var_noisy_coords.sqrt(), noisy_coords)
        struct_noise_scaled = 0.25 * torch.log(noise_level)
        noise_cond = self.noise_block(struct_noise_scaled)

        # Prepare self- and crop-conditioning and concatenate along channels
        if struct_self_cond is None:
            struct_self_cond = torch.zeros_like(noisy_coords)
        if self.config.model.crop_conditional:
            if struct_crop_cond is None:
                struct_crop_cond = torch.zeros_like(noisy_coords)
            else:
                struct_crop_cond = struct_crop_cond / self.sigma_data
            emb = torch.cat([emb, struct_self_cond, struct_crop_cond], -1)
        else:
            emb = torch.cat([emb, struct_self_cond], -1)

        # Run neural network
        emb = self.net(emb, noise_cond, seq_mask=seq_mask, residue_index=residue_index)

        # Preconditioning from Karras et al.
        out_scale = noise_level * actual_var_data**0.5 / torch.sqrt(var_noisy_coords)
        skip_scale = actual_var_data / var_noisy_coords
        emb = emb * utils.expand(out_scale, emb)
        skip_info = noisy_coords * utils.expand(skip_scale, noisy_coords)
        denoised_coords_x0 = emb + skip_info

        # Don't use atom mask; denoise all atoms
        denoised_coords_x0 *= utils.expand(seq_mask, denoised_coords_x0)
        return denoised_coords_x0


class Protpardelle(nn.Module):
    """All-atom protein diffusion-based generative model.

    This class wraps a structure denoising network and a sequence prediction network
    to do structure/sequence co-design (for all-atom generation), or backbone generation.

    It can be trained for one of four main tasks. To produce the all-atom (co-design)
    Protpardelle model, we will typically pretrain an 'allatom' model, then use this
    to train a 'seqdes' model. A 'seqdes' model can be trained with either a backbone
    or allatom denoiser. The two can be combined to yield all-atom (co-design) Protpardelle
    without further training.
        'backbone': train only a backbone coords denoiser.
        'seqdes': train only a mini-MPNN, using a pretrained coords denoiser.
        'allatom': train only an allatom coords denoiser (cannot do all-atom generation
            by itself).
        'codesign': train both an allatom denoiser and mini-MPNN at once.

    """

    def __init__(self, config: argparse.Namespace, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        self.task = config.model.task
        self.n_tokens = config.data.n_aatype_tokens

        self.use_mpnn_model = self.task in ["seqdes", "codesign"]

        # Modules
        self.all_modules = {}
        self.bb_idxs = [0, 1, 2, 4]
        self.n_atoms = 37
        self.struct_model = CoordinateDenoiser(config)
        self.all_modules["struct_model"] = self.struct_model
        self.bb_idxs = self.struct_model.bb_idxs
        self.n_atoms = self.struct_model.n_atoms

        if self.use_mpnn_model:
            self.mpnn_model = MiniMPNN(config)
            self.all_modules["mpnn_model"] = self.mpnn_model

        # Load any pretrained modules
        for module_name in self.config.model.pretrained_modules:
            self.load_pretrained_module(module_name)

        # Diffusion-related
        self.sigma_data = self.struct_model.sigma_data
        self.training_noise_schedule = partial(
            diffusion.noise_schedule,
            sigma_data=self.sigma_data,
            **vars(config.diffusion.training),
        )
        self.sampling_noise_schedule_default = self.make_sampling_noise_schedule()

    def load_pretrained_module(self, module_name: str, ckpt_path: Optional[str] = None):
        """Load pretrained weights for a given module name."""
        assert module_name in ["struct_model", "mpnn_model"], module_name

        # Load pretrained checkpoint
        if ckpt_path is None:
            ckpt_path = getattr(self.config.model, f"{module_name}_checkpoint")
            ckpt_path = os.path.join(self.config.train.home_dir, ckpt_path)
        ckpt_dict = torch.load(ckpt_path, map_location=self.device)
        model_state_dict = ckpt_dict["model_state_dict"]

        # Get only submodule state_dict
        submodule_state_dict = {
            sk[len(module_name) + 1 :]: sv
            for sk, sv in model_state_dict.items()
            if sk.startswith(module_name)
        }

        # Load into module
        module = dict(self.named_modules())[module_name]
        module.load_state_dict(submodule_state_dict)

        # Freeze unneeded modules
        if module_name == "struct_model":
            self.struct_model = module
            if self.task == "seqdes":
                for p in module.parameters():
                    p.requires_grad = False
        if module_name == "mpnn_model":
            self.mpnn_model = module
            if self.task not in ["codesign", "seqdes"]:
                for p in module.parameters():
                    p.requires_grad = False

        return module

    def load_minimpnn(self, mpnn_ckpt_path: Optional[str] = None):
        """Convert an allatom model to a codesign model."""
        if mpnn_ckpt_path is None:
            mpnn_ckpt_path = "checkpoints/minimpnn_state_dict.pth"
        self.mpnn_model = MiniMPNN(self.config).to(self.device)
        self.load_pretrained_module("mpnn_model", ckpt_path=mpnn_ckpt_path)
        self.use_mpnn_model = True
        return

    def remove_minimpnn(self):
        """Revert a codesign model to an allatom model to a codesign model."""
        self.use_mpnn_model = False
        self.mpnn_model = None
        self.all_modules["mpnn_model"] = None

    def make_sampling_noise_schedule(self, **noise_kwargs):
        """Make the default sampling noise schedule function."""
        noise_schedule_kwargs = vars(self.config.diffusion.sampling)
        if len(noise_kwargs) > 0:
            noise_schedule_kwargs.update(noise_kwargs)
        return partial(diffusion.noise_schedule, **noise_schedule_kwargs)

    def forward(
        self,
        *,
        noisy_coords: TensorType["b n a x", float],
        noise_level: TensorType["b", float],
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
        struct_self_cond: Optional[TensorType["b n a x", float]] = None,
        struct_crop_cond: Optional[TensorType["b n a x", float]] = None,
        seq_self_cond: Optional[TensorType["b n t", float]] = None,  # logprobs
        run_struct_model: bool = True,
        run_mpnn_model: bool = True,
    ):
        """Main forward function for denoising/co-design.

        Arguments:
            noisy_coords: noisy array of xyz coordinates.
            noise_level: std of noise for each example in the batch.
            seq_mask: mask indicating which indexes contain data.
            residue_index: residue ordering. This is used by proteinMPNN, but currently
                only used by the diffusion model when the 'absolute_residx' or
                'relative' position_embedding_type is specified.
            struct_self_cond: denoised coordinates from the previous step, scaled
                down by sigma data.
            struct_crop_cond: unnoised coordinates. unscaled (scaled down by sigma
                data inside the denoiser)
            seq_self_cond: mpnn-predicted sequence logprobs from the previous step.
            run_struct_model: flag to optionally not run structure denoiser.
            run_mpnn_model: flag to optionally not run mini-mpnn.
        """

        # Coordinate denoiser
        denoised_x0 = noisy_coords
        if run_struct_model:
            denoised_x0 = self.struct_model(
                noisy_coords,
                noise_level,
                seq_mask,
                residue_index=residue_index,
                struct_self_cond=struct_self_cond,
                struct_crop_cond=struct_crop_cond,
            )

        # Mini-MPNN
        aatype_logprobs = None
        if self.use_mpnn_model and run_mpnn_model:
            aatype_logprobs = self.mpnn_model(
                denoised_x0.detach(),
                noise_level,
                seq_mask,
                residue_index,
                seq_self_cond=seq_self_cond,
                return_embeddings=False,
            )
            aatype_logprobs = aatype_logprobs * seq_mask[..., None]

        # Process outputs
        if aatype_logprobs is None:
            aatype_logprobs = repeat(seq_mask, "b n -> b n t", t=self.n_tokens)
            aatype_logprobs = torch.ones_like(aatype_logprobs)
            aatype_logprobs = F.log_softmax(aatype_logprobs, -1)
        struct_self_cond_out = denoised_x0.detach() / self.sigma_data
        seq_self_cond_out = aatype_logprobs.detach()

        return denoised_x0, aatype_logprobs, struct_self_cond_out, seq_self_cond_out

    def make_seq_mask_for_sampling(
        self,
        prot_lens: Optional[TensorType["b", int]] = None,
        n_samples: int = 1,
        min_len: int = 50,
        max_len: Optional[int] = None,
    ):
        """Makes a sequence mask of varying protein lengths (only input required
        to begin sampling).
        """
        if max_len is None:
            max_len = self.config.data.fixed_size
        if prot_lens is None:
            possible_lens = np.arange(min_len, max_len)
            prot_lens = torch.Tensor(np.random.choice(possible_lens, n_samples))
        else:
            n_samples = len(prot_lens)
            max_len = max(prot_lens)
        mask = repeat(torch.arange(max_len), "n -> b n", b=n_samples)
        mask = (mask < prot_lens[:, None]).float().to(self.device)
        return mask

    def sample(
        self,
        *,
        seq_mask: TensorType["b n", float] = None,
        n_samples: int = 1,
        min_len: int = 50,
        max_len: int = 512,
        residue_index: TensorType["b n", int] = None,
        gt_coords: TensorType["b n a x", float] = None,
        gt_coords_traj: List[TensorType["b n a x", float]] = None,
        gt_cond_atom_mask: TensorType["b n a", float] = None,
        gt_aatype: TensorType["b n", int] = None,
        gt_cond_seq_mask: TensorType["b n", float] = None,
        apply_cond_proportion: float = 1.0,
        n_steps: int = 200,
        step_scale: float = 1.2,
        s_churn: float = 50.0,
        noise_scale: float = 1.0,
        s_t_min: float = 0.01,
        s_t_max: float = 50.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        disallow_aas: List[int] = [4, 20],  # cys, unk
        sidechain_mode: bool = False,
        skip_mpnn_proportion: float = 0.7,
        anneal_seq_resampling_rate: Optional[str] = None,  # linear, cosine
        use_fullmpnn: bool = False,
        use_fullmpnn_for_final: bool = True,
        use_reconstruction_guidance: bool = False,
        use_classifier_free_guidance: bool = False,  # defaults to replacement guidance if these are all false
        guidance_scale: float = 1.0,
        noise_schedule: Optional[Callable] = None,
        tqdm_pbar: Optional[Callable] = None,
        return_last: bool = True,
        return_aux: bool = False,
    ):
        """Sampling function for backbone or all-atom diffusion. All arguments are optional.

        Arguments:
            seq_mask: mask defining the number and lengths of proteins to be sampled.
            n_samples: number of samples to draw (if seq_mask not provided).
            min_len: minimum length of proteins to be sampled (if seq_mask not provided).
            max_len: maximum length of proteins to be sampled (if seq_mask not provided).
            residue_index: residue index of proteins to be sampled.
            gt_coords: conditioning information for coords.
            gt_coords_traj: conditioning information for coords specified for each timestep
                (if gt_coords is not provided).
            gt_cond_atom_mask: mask identifying atoms to apply gt_coords.
            gt_aatype: conditioning information for sequence.
            gt_cond_seq_mask: sequence positions to apply gt_aatype.
            apply_cond_proportion: the proportion of timesteps to apply the conditioning.
                e.g. if 0.5, then the first 50% of steps use conditioning, and the last 50%
                are unconditional.
            n_steps: number of denoising steps (ODE discretizations).
            step_scale: scale to apply to the score.
            s_churn: gamma = s_churn / n_steps describes the additional noise to add
                relatively at each denoising step. Use 0.0 for deterministic sampling or
                0.2 * n_steps as a rough default for stochastic sampling.
            noise_scale: scale to apply to gamma.
            s_t_min: don't apply s_churn below this noise level.
            s_t_max: don't apply s_churn above this noise level.
            temperature: scale to apply to aatype logits.
            top_p: don't tokens which fall outside this proportion of the total probability.
            disallow_aas: don't sample these token indices.
            sidechain_mode: whether to do all-atom sampling (False for backbone-only).
            skip_mpnn_proportion: proportion of timesteps from the start to skip running
                mini-MPNN.
            anneal_seq_resampling_rate: whether and how to decay the probability of
                running mini-MPNN. None, 'linear', or 'cosine'
            use_fullmpnn: use "full" ProteinMPNN at each step.
            use_fullmpnn_for_final: use "full" ProteinMPNN at the final step.
            use_reconstruction_guidance: use reconstruction guidance on the conditioning.
            use_classifier_free_guidance: use classifier-free guidance on the conditioning.
            guidance_scale: weight for reconstruction/classifier-free guidance.
            noise_schedule: specify the noise level timesteps for sampling.
            tqdm_pbar: progress bar in interactive contexts.
            return_last: return only the sampled structure and sequence.
            return_aux: return a dict of everything associated with the sampling run.
        """

        def ode_step(sigma_in, sigma_next, xt_in, x0_pred, gamma, guidance_in=None):
            if gamma > 0:
                t_hat = sigma_in + gamma * sigma_in
                sigma_delta = torch.sqrt(t_hat**2 - sigma_in**2)
                noisier_x = xt_in + utils.expand(
                    sigma_delta, xt_in
                ) * noise_scale * torch.randn_like(xt_in).to(xt_in)
                xt_in = noisier_x * utils.expand(seq_mask, noisier_x)
                sigma_in = t_hat

            mask = (sigma_in > 0).float()
            score = (xt_in - x0_pred) / utils.expand(sigma_in.clamp(min=1e-6), xt_in)
            score = score * utils.expand(mask, score)
            if use_reconstruction_guidance:
                guidance, guidance_mask = guidance_in
                guidance = guidance * guidance_mask[..., None]
                guidance_std = guidance[guidance_mask.bool()].var().sqrt()
                score_std = score[guidance_mask.bool()].var().sqrt()
                score = score + guidance * guidance_scale
            if use_classifier_free_guidance:
                # guidance_in is the unconditional x0 (x0_pred is the conditional x0)
                # guidance_scale = 1 + w from Ho paper
                # ==0: use only unconditional score; <1: interpolate the scores;
                # ==1: use only conditional score; >1: skew towards conditional score
                uncond_x0 = guidance_in
                uncond_score = (xt_in - uncond_x0) / utils.expand(
                    sigma_in.clamp(min=1e-6), xt_in
                )
                uncond_score = uncond_score * utils.expand(mask, uncond_score)
                score = guidance_scale * score + (1 - guidance_scale) * uncond_score
            step = score * step_scale * utils.expand(sigma_next - sigma_in, score)
            new_xt = xt_in + step
            return new_xt

        def sample_aatype(logprobs):
            # Top-p truncation
            probs = F.softmax(logprobs.clone(), dim=-1)
            sorted_prob, sorted_idxs = torch.sort(probs, descending=True)
            cumsum_prob = torch.cumsum(sorted_prob, dim=-1)
            sorted_indices_to_remove = cumsum_prob > top_p
            sorted_indices_to_remove[..., 0] = 0
            sorted_prob[sorted_indices_to_remove] = 0
            orig_probs = torch.scatter(
                torch.zeros_like(sorted_prob),
                dim=-1,
                index=sorted_idxs,
                src=sorted_prob,
            )

            # Apply temperature and disallowed AAs and sample
            assert temperature >= 0.0
            scaled_logits = orig_probs.clamp(min=1e-9).log() / (temperature + 1e-4)
            if disallow_aas:
                unwanted_mask = torch.zeros(scaled_logits.shape[-1]).to(scaled_logits)
                unwanted_mask[disallow_aas] = 1
                scaled_logits -= unwanted_mask * 1e10
            orig_probs = F.softmax(scaled_logits, dim=-1)
            categorical = torch.distributions.Categorical(probs=orig_probs)
            samp_aatype = categorical.sample()
            return samp_aatype

        def design_with_fullmpnn(batched_coords, seq_mask):
            seq_lens = seq_mask.sum(-1).long()
            designed_seqs = [
                evaluation.design_sequence(c[: seq_lens[i]], model=fullmpnn_model)[0]
                for i, c in enumerate(batched_coords)
            ]
            designed_aatypes, _ = utils.batched_seq_to_aatype_and_mask(
                designed_seqs, max_len=seq_mask.shape[-1]
            )
            return designed_aatypes

        # Initialize masks/features
        if seq_mask is None:  # Sample random lengths
            assert gt_aatype is None  # Don't condition on aatype without seq_mask
            seq_mask = self.make_seq_mask_for_sampling(
                n_samples=n_samples,
                min_len=min_len,
                max_len=max_len,
            )
        if residue_index is None:
            residue_index = torch.arange(seq_mask.shape[-1])
            residue_index = repeat(residue_index, "n -> b n", b=seq_mask.shape[0])
            residue_index = residue_index.to(seq_mask) * seq_mask
        if use_fullmpnn or use_fullmpnn_for_final:
            fullmpnn_model = protein_mpnn.get_mpnn_model(
                path_to_model_weights=self.config.train.home_dir
                + "/ProteinMPNN/vanilla_model_weights",
                device=self.device,
            )

        # Initialize noise schedule/parameters
        to_batch_size = lambda x: x * torch.ones(seq_mask.shape[0]).to(self.device)
        s_t_min = s_t_min * self.sigma_data
        s_t_max = s_t_max * self.sigma_data
        if noise_schedule is None:
            noise_schedule = self.sampling_noise_schedule_default
        sigma = noise_schedule(1)
        timesteps = torch.linspace(1, 0, n_steps + 1)

        # Set up conditioning/guidance information
        crop_cond_coords = None
        if gt_coords is None:
            coords_shape = seq_mask.shape + (self.n_atoms, 3)
            xt = torch.randn(*coords_shape).to(self.device) * sigma
            xt *= utils.expand(seq_mask, xt)
        else:
            assert gt_coords_traj is None
            noise_levels = [to_batch_size(noise_schedule(t)) for t in timesteps]
            gt_coords_traj = [
                diffusion.noise_coords(gt_coords, nl) for nl in noise_levels
            ]
            xt = gt_coords_traj[0]
            if gt_cond_atom_mask is not None:
                crop_cond_coords = gt_coords * gt_cond_atom_mask[..., None]
        gt_atom_mask = None
        if gt_aatype is not None:
            gt_atom_mask = utils.atom37_mask_from_aatype(gt_aatype, seq_mask)
        fake_logits = repeat(seq_mask, "b n -> b n t", t=self.n_tokens)
        s_hat = (sample_aatype(fake_logits) * seq_mask).long()

        # Initialize superposition for all-atom sampling
        if sidechain_mode:
            b, n = seq_mask.shape[:2]

            # Latest predicted x0 for sidechain superpositions
            atom73_state_0 = torch.zeros(b, n, 73, 3).to(xt)

            # Current state xt for sidechain superpositions (denoised to different levels)
            atom73_state_t = torch.randn(b, n, 73, 3).to(xt) * sigma

            # Noise level of xt
            sigma73_last = torch.ones(b, n, 73).to(xt) * sigma

            # Seqhat and mask used to choose sidechains for euler step (b, n)
            s_hat = (seq_mask * 7).long()
            mask37 = utils.atom37_mask_from_aatype(s_hat, seq_mask).bool()
            mask73 = utils.atom73_mask_from_aatype(s_hat, seq_mask).bool()
            begin_mpnn_step = int(n_steps * skip_mpnn_proportion)

        # Prepare to run sampling trajectory
        sigma = to_batch_size(sigma)
        x0 = None
        x0_prev = None
        x_self_cond = None
        s_logprobs = None
        s_self_cond = None
        if tqdm_pbar is None:
            tqdm_pbar = lambda x: x
        torch.set_grad_enabled(False)

        # *t_traj is the denoising trajectory; *0_traj is the evolution of predicted clean data
        # s0 are aatype probs of shape (b n t); s_hat are discrete aatype of shape (b n)
        xt_traj, x0_traj, st_traj, s0_traj = [], [], [], []

        # Sampling trajectory
        for i, t in tqdm_pbar(enumerate(iter(timesteps[1:]))):
            # Set up noise levels
            sigma_next = noise_schedule(t)
            if i == n_steps - 1:
                sigma_next *= 0
            gamma = (
                s_churn / n_steps
                if (sigma_next >= s_t_min and sigma_next <= s_t_max)
                else 0.0
            )
            sigma_next = to_batch_size(sigma_next)

            if sidechain_mode:
                # Fill in noise for masked positions since xt is initialized to zeros at each step
                dummy_fill_noise = torch.randn_like(xt) * utils.expand(sigma, xt)
                zero_atom_mask = utils.atom37_mask_from_aatype(s_hat, seq_mask)
                dummy_fill_mask = 1 - zero_atom_mask[..., None]
                xt = xt * zero_atom_mask[..., None] + dummy_fill_noise * dummy_fill_mask
            else:  # backbone only
                bb_seq = (seq_mask * residue_constants.restype_order["G"]).long()
                bb_atom_mask = utils.atom37_mask_from_aatype(bb_seq, seq_mask)
                xt *= bb_atom_mask[..., None]

            # Enable grad for reconstruction guidance
            if use_reconstruction_guidance:
                torch.set_grad_enabled(True)
                xt.requires_grad = True

            # Run denoising network
            run_mpnn = not sidechain_mode or i > begin_mpnn_step
            x0, s_logprobs, x_self_cond, s_self_cond = self.forward(
                noisy_coords=xt,
                noise_level=sigma,
                seq_mask=seq_mask,
                residue_index=residue_index,
                struct_self_cond=x_self_cond,
                struct_crop_cond=crop_cond_coords,
                seq_self_cond=s_self_cond,
                run_mpnn_model=run_mpnn,
            )

            # Compute additional stuff for guidance
            if use_reconstruction_guidance:
                loss = (x0 - gt_coords).pow(2).sum(-1)
                loss = loss * gt_cond_atom_mask
                loss = loss.sum() / gt_cond_atom_mask.sum().clamp(min=1)
                xt.retain_grad()
                loss.backward()
                guidance = xt.grad.clone()
                xt.grad *= 0
                torch.set_grad_enabled(False)
            if use_classifier_free_guidance:
                assert not use_reconstruction_guidance
                uncond_x0, _, _, _ = self.forward(
                    noisy_coords=xt,
                    noise_level=sigma,
                    seq_mask=seq_mask,
                    residue_index=residue_index,
                    struct_self_cond=x_self_cond,
                    seq_self_cond=s_self_cond,
                    run_mpnn_model=run_mpnn,
                )

            # Structure denoising step
            if not sidechain_mode:  # backbone
                if sigma[0] > 0:
                    xt = ode_step(sigma, sigma_next, xt, x0, gamma)
                else:
                    xt = x0
            else:  # allatom
                # Write x0 into atom73_state_0 for atoms corresponding to old seqhat
                atom73_state_0[mask73] = x0[mask37]

                # Determine sequence resampling probability
                if anneal_seq_resampling_rate is not None:
                    step_time = 1 - (i - begin_mpnn_step) / max(
                        1, n_steps - begin_mpnn_step
                    )
                    if anneal_seq_resampling_rate == "linear":
                        resampling_rate = step_time
                    elif anneal_seq_resampling_rate == "cosine":
                        k = 2
                        resampling_rate = (
                            1 + np.cos(2 * np.pi * (step_time - 0.5))
                        ) / k
                    resample_this_step = np.random.uniform() < resampling_rate

                # Resample sequence or design with full ProteinMPNN
                if i == n_steps - 1 and use_fullmpnn_for_final:
                    s_hat = design_with_fullmpnn(x0, seq_mask).to(x0.device)
                elif anneal_seq_resampling_rate is None or resample_this_step:
                    if run_mpnn and use_fullmpnn:
                        s_hat = design_with_fullmpnn(x0, seq_mask).to(x0.device)
                    else:
                        s_hat = sample_aatype(s_logprobs)

                # Overwrite s_hat with any conditioning information
                if (i + 1) / n_steps <= apply_cond_proportion:
                    if gt_cond_seq_mask is not None and gt_aatype is not None:
                        s_hat = (
                            1 - gt_cond_seq_mask
                        ) * s_hat + gt_cond_seq_mask * gt_aatype
                        s_hat = s_hat.long()

                # Set masks for collapsing superposition using new sequence
                mask37 = utils.atom37_mask_from_aatype(s_hat, seq_mask).bool()
                mask73 = utils.atom73_mask_from_aatype(s_hat, seq_mask).bool()

                # Determine prev noise levels for atoms corresponding to new sequence
                step_sigma_prev = (
                    torch.ones(*xt.shape[:-1]).to(xt) * sigma[..., None, None]
                )
                step_sigma_prev[mask37] = sigma73_last[mask73]  # b, n, 37
                step_sigma_next = sigma_next[..., None, None]  # b, 1, 1

                # Denoising step on atoms corresponding to new sequence
                b, n = mask37.shape[:2]
                step_xt = torch.zeros(b, n, 37, 3).to(xt)
                step_x0 = torch.zeros(b, n, 37, 3).to(xt)
                step_xt[mask37] = atom73_state_t[mask73]
                step_x0[mask37] = atom73_state_0[mask73]

                guidance_in = None
                if (i + 1) / n_steps <= apply_cond_proportion:
                    if use_reconstruction_guidance:
                        guidance_in = (guidance, mask37.float())
                    elif use_classifier_free_guidance:
                        guidance_in = uncond_x0

                step_xt = ode_step(
                    step_sigma_prev,
                    step_sigma_next,
                    step_xt,
                    step_x0,
                    gamma,
                    guidance_in=guidance_in,
                )
                xt = step_xt

                # Write new xt into atom73_state_t for atoms corresponding to new seqhat and update sigma_last
                atom73_state_t[mask73] = step_xt[mask37]
                sigma73_last[mask73] = step_sigma_next[0].item()

            # Replacement guidance if conditioning information provided
            if (i + 1) / n_steps <= apply_cond_proportion:
                if gt_coords_traj is not None:
                    if gt_cond_atom_mask is None:
                        xt = gt_coords_traj[i + 1]
                    else:
                        xt = (1 - gt_cond_atom_mask)[
                            ..., None
                        ] * xt + gt_cond_atom_mask[..., None] * gt_coords_traj[i + 1]

            sigma = sigma_next

            # Logging
            xt_scale = self.sigma_data / utils.expand(
                torch.sqrt(sigma_next**2 + self.sigma_data**2), xt
            )
            scaled_xt = xt * xt_scale
            xt_traj.append(scaled_xt.cpu())
            x0_traj.append(x0.cpu())
            st_traj.append(s_hat.cpu())
            s0_traj.append(s_logprobs.cpu())

        if return_last:
            return xt, s_hat, seq_mask
        elif return_aux:
            return {
                "x": xt,
                "s": s_hat,
                "seq_mask": seq_mask,
                "xt_traj": xt_traj,
                "x0_traj": x0_traj,
                "st_traj": st_traj,
                "s0_traj": s0_traj,
            }
        else:
            return xt_traj, x0_traj, st_traj, s0_traj, seq_mask
