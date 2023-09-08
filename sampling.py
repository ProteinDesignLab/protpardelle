"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Configs and convenience functions for wrapping the model sample() function.
"""
import argparse
import time
from typing import Optional, Tuple

import torch
from torchtyping import TensorType

from core import residue_constants
from core import utils
import diffusion


def default_backbone_sampling_config():
    config = argparse.Namespace(
        n_steps=500,
        s_churn=200,
        step_scale=1.2,
        sidechain_mode=False,
        noise_schedule=lambda t: diffusion.noise_schedule(t, s_max=80, s_min=0.001),
    )
    return config


def default_allatom_sampling_config():
    noise_schedule = lambda t: diffusion.noise_schedule(t, s_max=80, s_min=0.001)
    stage2 = argparse.Namespace(
        apply_cond_proportion=1.0,
        n_steps=200,
        s_churn=100,
        step_scale=1.2,
        sidechain_mode=True,
        skip_mpnn_proportion=1.0,
        noise_schedule=noise_schedule,
    )
    config = argparse.Namespace(
        n_steps=500,
        s_churn=200,
        step_scale=1.2,
        sidechain_mode=True,
        skip_mpnn_proportion=0.6,
        use_fullmpnn=False,
        use_fullmpnn_for_final=True,
        anneal_seq_resampling_rate="linear",
        noise_schedule=noise_schedule,
        stage_2=stage2,
    )
    return config


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
