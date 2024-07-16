"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Configs and convenience functions for wrapping the model sample() function.
Utils for forward ODE for likelihoods/encoding.
"""

import argparse
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torchtyping import TensorType
from tqdm import tqdm

from core import data
from core import residue_constants
from core import utils
import diffusion


def draw_backbone_samples(
    model: torch.nn.Module,
    seq_mask: TensorType["b n", float] = None,
    n_samples: int = None,
    sample_length_range: Tuple[int] = (50, 512),
    pdb_save_path: Optional[str] = None,
    return_aux: bool = False,
    return_sampling_runtime: bool = False,
    **sampling_kwargs,
):
    device = model.device
    if seq_mask is None:
        assert n_samples is not None
        seq_mask = model.make_seq_mask_for_sampling(
            n_samples=n_samples,
            min_len=sample_length_range[0],
            max_len=sample_length_range[1],
        )

    start = time.time()
    aux = model.sample(
        seq_mask=seq_mask, return_last=False, return_aux=True, **sampling_kwargs
    )
    aux["runtime"] = time.time() - start
    seq_lens = seq_mask.sum(-1).long()
    cropped_samp_coords = [
        s[: seq_lens[i], model.bb_idxs] for i, s in enumerate(aux["xt_traj"][-1])
    ]

    if pdb_save_path is not None:
        gly_aatype = (seq_mask * residue_constants.restype_order["G"]).long()
        trimmed_aatype = [a[: seq_lens[i]] for i, a in enumerate(gly_aatype)]
        atom_mask = utils.atom37_mask_from_aatype(gly_aatype, seq_mask).cpu()

        for i in range(len(cropped_samp_coords)):
            utils.write_coords_to_pdb(
                cropped_samp_coords[i],
                f"{pdb_save_path}{i}.pdb",
                batched=False,
                aatype=trimmed_aatype[i],
                atom_mask=atom_mask[i],
            )

    if return_aux:
        return aux
    else:
        if return_sampling_runtime:
            return cropped_samp_coords, seq_mask, aux["runtime"]
        else:
            return cropped_samp_coords, seq_mask


def draw_allatom_samples(
    model: torch.nn.Module,
    seq_mask: TensorType["b n", float] = None,
    n_samples: int = None,
    sample_length_range: Tuple[int] = (50, 512),
    two_stage_sampling: bool = True,
    pdb_save_path: Optional[str] = None,
    return_aux: bool = False,
    return_sampling_runtime: bool = False,
    **sampling_kwargs,
):
    """Implement the default 2-stage all-atom sampling routine."""

    def save_allatom_samples(aux, path):
        seq_lens = aux["seq_mask"].sum(-1).long()
        cropped_samp_coords = [
            c[: seq_lens[i]] for i, c in enumerate(aux["xt_traj"][-1])
        ]
        cropped_samp_aatypes = [
            s[: seq_lens[i]] for i, s in enumerate(aux["st_traj"][-1])
        ]
        samp_atom_mask = utils.atom37_mask_from_aatype(
            aux["st_traj"][-1].to(device), seq_mask
        )
        samp_atom_mask = [m[: seq_lens[i]] for i, m in enumerate(samp_atom_mask)]
        for i, c in enumerate(cropped_samp_coords):
            utils.write_coords_to_pdb(
                c,
                f"{path}{i}.pdb",
                batched=False,
                aatype=cropped_samp_aatypes[i],
                atom_mask=samp_atom_mask[i],
                conect=True,
            )

    device = model.device
    if seq_mask is None:
        assert n_samples is not None
        seq_mask = model.make_seq_mask_for_sampling(
            n_samples=n_samples,
            min_len=sample_length_range[0],
            max_len=sample_length_range[1],
        )
    sampling_runtime = 0.0

    # Stage 1 sampling
    start = time.time()
    if "stage_2" in sampling_kwargs:
        stage_2_kwargs = vars(sampling_kwargs.pop("stage_2"))
    aux = model.sample(
        seq_mask=seq_mask,
        return_last=False,
        return_aux=True,
        **sampling_kwargs,
    )
    sampling_runtime = time.time() - start
    if pdb_save_path is not None and two_stage_sampling:
        save_allatom_samples(aux, pdb_save_path + "_init")

    # Stage 2 sampling (sidechain refinement only)
    if two_stage_sampling:
        samp_seq = aux["st_traj"][-1]
        samp_coords = aux["xt_traj"][-1]
        cond_atom_mask = utils.atom37_mask_from_aatype((seq_mask * 7).long(), seq_mask)
        aux = {f"stage1_{k}": v for k, v in aux.items()}
        start = time.time()
        stage2_aux = model.sample(
            gt_cond_atom_mask=cond_atom_mask.to(device),  # condition on backbone
            gt_cond_seq_mask=seq_mask.to(device),
            gt_coords=samp_coords.to(device),
            gt_aatype=samp_seq.to(device),
            seq_mask=seq_mask,
            return_last=False,
            return_aux=True,
            **stage_2_kwargs,
        )
        sampling_runtime += time.time() - start
        aux = {**aux, **stage2_aux}
    if pdb_save_path is not None:
        save_allatom_samples(aux, pdb_save_path + "_samp")
    aux["runtime"] = sampling_runtime

    # Process outputs, crop to correct length
    if return_aux:
        return aux
    else:
        xt_traj = aux["xt_traj"]
        st_traj = aux["st_traj"]
        seq_mask = aux["seq_mask"]
        seq_lens = seq_mask.sum(-1).long()
        cropped_samp_coords = [c[: seq_lens[i]] for i, c in enumerate(xt_traj[-1])]
        cropped_samp_aatypes = [s[: seq_lens[i]] for i, s in enumerate(st_traj[-1])]
        samp_atom_mask = utils.atom37_mask_from_aatype(st_traj[-1].to(device), seq_mask)
        samp_atom_mask = [m[: seq_lens[i]] for i, m in enumerate(samp_atom_mask)]
        orig_xt_traj = aux["stage1_xt_traj"]
        stage1_coords = [c[: seq_lens[i]] for i, c in enumerate(orig_xt_traj[-1])]
        ret = (
            cropped_samp_coords,
            cropped_samp_aatypes,
            samp_atom_mask,
            stage1_coords,
            seq_mask,
        )
        if return_sampling_runtime:
            ret = ret + (sampling_runtime,)
        return ret


def get_backbone_mask(atom_mask):
    backbone_mask = torch.zeros_like(atom_mask)
    for atom in ("N", "CA", "C", "O"):
        backbone_mask[..., residue_constants.atom_order[atom]] = 1
    return backbone_mask


def batch_from_pdbs(list_of_pdbs, seed=None):
    all_feats = []
    for pdb in list_of_pdbs:
        if "1qjg" in pdb and False:
            all_feats.append(
                utils.load_feats_from_pdb(pdb, chain_id="A", protein_only=True)
            )
        else:
            all_feats.append(utils.load_feats_from_pdb(pdb))
    max_len = max([f["aatype"].shape[0] for f in all_feats])
    dict_of_lists = {"seq_mask": []}
    for feats in all_feats:
        for k, v in feats.items():
            if k in ["atom_mask", "atom_positions", "residue_index"]:
                if k == "atom_positions":
                    v = data.apply_random_se3(
                        v, atom_mask=feats["atom_mask"], translation_scale=0, seed=seed
                    )
                padded_feat, seq_mask = data.make_fixed_size_1d(v, max_len)
                dict_of_lists.setdefault(k, []).append(padded_feat)
        dict_of_lists["seq_mask"].append(seq_mask)
    return {k: torch.stack(v) for k, v in dict_of_lists.items()}


def forward_ode(
    model,
    batch,
    n_steps=100,
    sigma_min=0.01,
    sigma_max=800,
    tqdm_pbar=None,
    seed=0,
    verbose=False,
    eps=None,
):
    """Solve the probability flow ODE to get latent encodings and likelihoods.

    Usage: given a backbone model `model` and a list of pdb paths `paths`
    batch = batch_from_pdbs(paths)
    results = forward_ode(model, batch)
    nats_per_atom = results['npa']
    latents = results['encoded_latent']

    Based on https://github.com/yang-song/score_sde_pytorch/blob/main/likelihood.py
    See also https://github.com/crowsonkb/k-diffusion/blob/cc49cf6182284e577e896943f8e29c7c9d1a7f2c/k_diffusion/sampling.py#L281
    """
    assert model.task == "backbone"
    device = model.device
    sigma_data = model.sigma_data
    torch.manual_seed(seed)

    seq_mask = batch["seq_mask"].to(device)
    to_batch_size = lambda x: x * torch.ones(seq_mask.shape[0]).to(device)
    residue_index = batch["residue_index"].to(device)
    backbone_mask = get_backbone_mask(batch["atom_mask"]) * batch["atom_mask"]
    backbone_mask = torch.ones_like(batch["atom_positions"]) * backbone_mask[..., None]
    init_bb_coords = (batch["atom_positions"] * backbone_mask).to(device)
    backbone_mask = backbone_mask.to(device)
    batch_data_sizes = backbone_mask.sum((1, 2, 3))

    # Noise for skilling-hutchinson
    if eps is None:
        eps = torch.randn_like(init_bb_coords)
    sum_dlogp = to_batch_size(0)

    # Initialize noise schedule/parameters
    noise_schedule = lambda t: diffusion.noise_schedule(
        t, s_min=sigma_min / sigma_data, s_max=sigma_max / sigma_data
    )
    timesteps = torch.linspace(0, 1, n_steps + 1)
    sigma = noise_schedule(timesteps[0])

    # init to sigma_min
    xt = init_bb_coords + torch.randn_like(init_bb_coords) * sigma

    sigma = to_batch_size(sigma)
    if tqdm_pbar is None:
        tqdm_pbar = lambda x: x
    xt_traj, x0_traj = [], []

    def dx_dt_f_theta(xt, sigma, sigma_next):
        xt = xt * backbone_mask
        x0, _, _, _ = model.forward(
            noisy_coords=xt,
            noise_level=sigma,
            seq_mask=seq_mask,
            residue_index=residue_index,
            run_mpnn_model=False,
        )
        dx_dt = (xt - x0) / utils.expand(sigma, xt)
        dx_dt = dx_dt * backbone_mask
        return dx_dt

    # Forward PF ODE
    with torch.no_grad():
        for i, t in tqdm_pbar(enumerate(iter(timesteps[1:]))):
            sigma_next = noise_schedule(t)
            sigma_next = to_batch_size(sigma_next)
            step_size = sigma_next - sigma

            # Euler integrator
            with torch.enable_grad():
                xt.requires_grad_(True)
                dx_dt = dx_dt_f_theta(xt, sigma, sigma_next)
                hutch_proj = (dx_dt * eps * backbone_mask).sum()
                grad = torch.autograd.grad(hutch_proj, xt)[0]
            xt.requires_grad_(False)
            dx = dx_dt * utils.expand(step_size, dx_dt)
            xt = xt + dx
            div = dlogp_dt = (grad * eps * backbone_mask).sum((1, 2, 3))
            dlogp = dlogp_dt * utils.expand(step_size, dlogp_dt)
            sum_dlogp = sum_dlogp + dlogp

            sigma = sigma_next

            # Logging
            xt_scale = sigma_data / utils.expand(
                torch.sqrt(sigma_next**2 + sigma_data**2), xt
            )
            scaled_xt = xt * xt_scale
            xt_traj.append(scaled_xt.cpu())

    prior_logp = -1 * batch_data_sizes / 2.0 * np.log(2 * np.pi * sigma_max**2) - (
        xt * xt
    ).sum((1, 2, 3)) / (2 * sigma_max**2)

    logp = prior_logp + sum_dlogp
    nats_per_atom = -logp / batch_data_sizes * 3
    bits_per_dim = -logp / batch_data_sizes / np.log(2)
    results = {
        "prior_logp": prior_logp,
        "prior_logp_per_atom": prior_logp / batch_data_sizes * 3,
        "deltalogp": sum_dlogp,
        "deltalogp_per_atom": sum_dlogp / batch_data_sizes * 3,
        "logp": logp,
        "npa": nats_per_atom,
        "bpd": bits_per_dim,
        "batch_data_sizes": batch_data_sizes,
        "protein_lengths": seq_mask.sum(-1),
        "encoded_latent": xt,
    }
    if verbose:
        for k, v in results.items():
            print(k, v)
    return results
