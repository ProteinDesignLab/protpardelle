"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Utils for computing evaluation metrics.
"""
import argparse
import os
import warnings
from typing import Tuple

from Bio.Align import substitution_matrices
import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from torchtyping import TensorType

from core import residue_constants
from core import utils
from core import protein_mpnn as mpnn
import modules
import sampling


def mean(x):
    if len(x) == 0:
        return 0
    return sum(x) / len(x)


def calculate_seq_identity(seq1, seq2, seq_mask=None):
    identity = (seq1 == seq2.to(seq1)).float()
    if seq_mask is not None:
        identity *= seq_mask.to(seq1)
        return identity.sum(-1) / seq_mask.to(seq1).sum(-1).clamp(min=1)
    else:
        return identity.mean(-1)


def design_sequence(coords, model=None, num_seqs=1, disallow_aas=["C"]):
    # Returns list of strs; seqs like 'MKRLLDS', not aatypes
    if model is None:
        model = mpnn.get_mpnn_model()
    if isinstance(coords, str):
        temp_pdb = False
        pdb_fn = coords
    else:
        temp_pdb = True
        pdb_fn = f"tmp{np.random.randint(0, 1e8)}.pdb"
        gly_idx = residue_constants.restype_order["G"]
        gly_aatype = (torch.ones(coords.shape[0]) * gly_idx).long()
        utils.write_coords_to_pdb(coords, pdb_fn, batched=False, aatype=gly_aatype)

    with torch.no_grad():
        designed_seqs = mpnn.run_proteinmpnn(
            model=model,
            pdb_path=pdb_fn,
            num_seq_per_target=num_seqs,
            omit_AAs=disallow_aas,
        )

    if temp_pdb:
        os.system("rm " + pdb_fn)
    return designed_seqs


def get_esmfold_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
    model.esm = model.esm.half()
    return model


def inference_esmfold(sequence_list, model, tokenizer):
    inputs = tokenizer(
        sequence_list,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)
    outputs = model(**inputs)
    # positions is shape (l, b, n, a, c)
    pred_coords = outputs.positions[-1].contiguous()
    plddts = (outputs.plddt[:, :, 1] * inputs.attention_mask).sum(
        -1
    ) / inputs.attention_mask.sum(-1).clamp(min=1e-3)
    return pred_coords, plddts


def predict_structures(sequences, model="esmfold", tokenizer=None, force_unk_to_X=True):
    # Expects seqs like 'MKRLLDS', not aatypes
    # model can be a model, or a string describing which pred model to load
    if isinstance(sequences, str):
        sequences = [sequences]
    if model == "esmfold":
        model = get_esmfold_model()
    device = model.device
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    aatype = [utils.seq_to_aatype(seq).to(device) for seq in sequences]

    with torch.no_grad():
        if isinstance(model, EsmForProteinFolding):
            pred_coords, plddts = inference_esmfold(sequences, model, tokenizer)

    seq_lens = [len(s) for s in sequences]
    trimmed_coords = [c[: seq_lens[i]] for i, c in enumerate(pred_coords)]
    trimmed_coords_atom37 = [
        utils.atom37_coords_from_atom14(c, aatype[i])
        for i, c in enumerate(trimmed_coords)
    ]
    return trimmed_coords_atom37, plddts


def compute_structure_metric(coords1, coords2, metric="ca_rmsd", atom_mask=None):
    # coords1 tensor[l][a][3]
    def _tmscore(a, b, mask=None):
        length = len(b)
        dists = (a - b).pow(2).sum(-1)
        d0 = 1.24 * ((length - 15) ** (1 / 3)) - 1.8
        term = 1 / (1 + ((dists) / (d0**2)))
        if mask is None:
            return term.mean()
        else:
            term = term * mask
            return term.sum() / mask.sum().clamp(min=1)

    aligned_coords1_ca, (R, t) = utils.kabsch_align(coords1[:, 1], coords2[:, 1])
    aligned_coords1 = coords1 - coords1[:, 1:2].mean(0, keepdim=True)
    aligned_coords1 = aligned_coords1 @ R.t() + t
    if metric == "ca_rmsd":
        return (aligned_coords1_ca - coords2[:, 1]).pow(2).sum(-1).sqrt().mean()
    elif metric == "tm_score":
        tm = _tmscore(aligned_coords1_ca, coords2[:, 1])
        # TODO: return 1 - tm score for now so sorts work properly
        return 1 - tm
    elif metric == "allatom_tm":
        # Align on Ca, compute allatom TM
        assert atom_mask is not None
        return _tmscore(aligned_coords1, coords2, mask=atom_mask)
    elif metric == "allatom_lddt":
        assert atom_mask is not None
        lddt = modules.lddt(
            coords1.reshape(-1, 3),
            coords2.reshape(-1, 3),
            atom_mask.reshape(-1, 1),
            per_residue=False,
        )
        return lddt
    else:
        raise NotImplementedError


def compute_self_consistency(
    comparison_structures,  # can be sampled or ground truth
    sampled_sequences=None,
    mpnn_model=None,
    struct_pred_model=None,
    tokenizer=None,
    num_seqs=1,
    return_aux=False,
    metric="ca_rmsd",
    output_file=None,
):
    # Typically used for eval of backbone sampling or sequence design or joint sampling
    # (Maybe MPNN) + Fold + TM/RMSD
    # Expects seqs like 'MKRLLDS', not aatypes
    per_sample_primary_metrics = []
    per_sample_secondary_metrics = []
    per_sample_plddts = []
    per_sample_coords = []
    per_sample_seqs = []
    aux = {}
    for i, coords in enumerate(comparison_structures):
        if sampled_sequences is None:
            seqs_to_predict = design_sequence(
                coords, model=mpnn_model, num_seqs=num_seqs
            )
        else:
            seqs_to_predict = sampled_sequences[i]
        pred_coords, plddts = predict_structures(
            seqs_to_predict, model=struct_pred_model, tokenizer=tokenizer
        )
        primary_metric_name = "tm_score" if metric == "tm_score" else "ca_rmsd"
        secondary_metric_name = "tm_score" if metric == "both" else None
        primary_metrics = [
            compute_structure_metric(coords.to(pred), pred, metric=primary_metric_name)
            for pred in pred_coords
        ]
        if secondary_metric_name:
            secondary_metrics = [
                compute_structure_metric(
                    coords.to(pred), pred, metric=secondary_metric_name
                )
                for pred in pred_coords
            ]
            aux.setdefault(secondary_metric_name, []).extend(secondary_metrics)
        else:
            secondary_metrics = primary_metrics

        aux.setdefault("pred", []).extend(pred_coords)
        seqs_to_predict_arr = seqs_to_predict
        if isinstance(seqs_to_predict_arr, str):
            seqs_to_predict_arr = [seqs_to_predict_arr]

        aux.setdefault("seqs", []).extend(seqs_to_predict_arr)
        aux.setdefault("plddt", []).extend(plddts)
        aux.setdefault("rmsd", []).extend(primary_metrics)

        # Report best rmsd design only among MPNN reps
        all_designs = [
            (m, p, t, c, s)
            for m, p, t, c, s in zip(
                primary_metrics,
                plddts,
                secondary_metrics,
                pred_coords,
                seqs_to_predict_arr,
            )
        ]
        best_rmsd_design = min(all_designs, key=lambda x: x[0])
        per_sample_primary_metrics.append(best_rmsd_design[0].detach().cpu())
        per_sample_plddts.append(best_rmsd_design[1].detach().cpu())
        per_sample_secondary_metrics.append(best_rmsd_design[2].detach().cpu())
        per_sample_coords.append(best_rmsd_design[3])
        per_sample_seqs.append(best_rmsd_design[4])
    best_idx = np.argmin(per_sample_primary_metrics)
    metrics = {
        "sc_rmsd_best": per_sample_primary_metrics[best_idx],
        "sc_plddt_best": per_sample_plddts[best_idx],
        "sc_rmsd_mean": mean(per_sample_primary_metrics),
        "sc_plddt_mean": mean(per_sample_plddts),
    }
    if metric == "both":
        metrics["sc_tmscore_best"] = per_sample_secondary_metrics[best_idx]
        metrics["sc_tmscore_mean"] = mean(per_sample_secondary_metrics)

    if output_file:
        pred_coords = per_sample_coords
        designed_seqs = per_sample_seqs

        if torch.isnan(pred_coords[best_idx]).sum() == 0:
            designed_seq = utils.seq_to_aatype(designed_seqs[best_idx])
            utils.write_coords_to_pdb(
                pred_coords[best_idx],
                output_file,
                batched=False,
                aatype=designed_seq,
            )

    if return_aux:
        return metrics, best_idx, aux
    else:
        return metrics, best_idx


def compute_secondary_structure_content(coords_batch):
    dssp_sample = []
    for i, c in enumerate(coords_batch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dssp_str = utils.get_3state_dssp(coords=c)
        if dssp_str is None or len(dssp_str) == 0:
            pass
        else:
            dssp_sample.append(dssp_str)
    dssp_sample = "".join(dssp_sample)
    metrics = {}
    metrics["sample_pct_beta"] = mean([c == "E" for c in dssp_sample])
    metrics["sample_pct_alpha"] = mean([c == "H" for c in dssp_sample])
    return metrics


def compute_bond_length_metric(
    cropped_coords_list, cropped_aatypes_list, atom_mask=None
):
    bond_length_dict = utils.batched_fullatom_bond_lengths_from_coords(
        cropped_coords_list, cropped_aatypes_list, atom_mask=atom_mask
    )
    all_errors = {}
    for aa1, d in bond_length_dict.items():
        aa3 = residue_constants.restype_1to3[aa1]
        per_bond_errors = []
        for bond, lengths in d.items():
            a1, a2 = bond.split("-")
            ideal_val = None
            for bond in residue_constants.standard_residue_bonds[aa3]:
                if (
                    bond.atom1_name == a1
                    and bond.atom2_name == a2
                    or bond.atom1_name == a2
                    and bond.atom2_name == a1
                ):
                    ideal_val = bond.length
                    break
            error = (np.array(lengths) - ideal_val) ** 2
            per_bond_errors.append(error.mean() ** 0.5)
        if len(per_bond_errors) > 0:  # often no Cys
            per_res_errors = np.mean(per_bond_errors)
            all_errors[aa1] = per_res_errors
    return np.mean(list(all_errors.values()))


def evaluate_backbone_generation(
    model,
    n_samples=1,
    mpnn_model=None,
    struct_pred_model=None,
    tokenizer=None,
    sample_length_range=(50, 512),
):
    sampling_config = sampling.default_backbone_sampling_config()
    trimmed_coords, seq_mask = sampling.draw_backbone_samples(
        model,
        n_samples=n_samples,
        sample_length_range=sample_length_range,
        **vars(sampling_config),
    )
    sc_metrics, best_idx, aux = compute_self_consistency(
        trimmed_coords,
        mpnn_model=mpnn_model,
        struct_pred_model=struct_pred_model,
        tokenizer=tokenizer,
        return_aux=True,
    )
    dssp_metrics = compute_secondary_structure_content(trimmed_coords)
    all_metrics = {**sc_metrics, **dssp_metrics}
    all_metrics = {f"bb_{k}": v for k, v in all_metrics.items()}
    return all_metrics, (trimmed_coords, seq_mask, best_idx, aux["pred"], aux["seqs"])


def evaluate_allatom_generation(
    model,
    n_samples,
    two_stage_sampling=True,
    struct_pred_model=None,
    tokenizer=None,
    sample_length_range=(50, 512),
):
    # Convert allatom model to codesign model by loading miniMPNN
    model.task = "codesign"
    model.load_minimpnn()
    model.eval()

    sampling_config = sampling.default_allatom_sampling_config()
    ret = sampling.draw_allatom_samples(
        model,
        n_samples=n_samples,
        two_stage_sampling=two_stage_sampling,
        **vars(sampling_config),
    )
    (
        cropped_samp_coords,
        cropped_samp_aatypes,
        samp_atom_mask,
        stage1_coords,
        seq_mask,
    ) = ret

    # Compute self consistency
    if struct_pred_model is None:
        struct_pred_model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1"
        ).to(device)
        struct_pred_model.esm = struct_pred_model.esm.half()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    designed_seqs = [utils.aatype_to_seq(a) for a in cropped_samp_aatypes]
    sc_metrics, best_idx, sc_aux = compute_self_consistency(
        comparison_structures=cropped_samp_coords,
        sampled_sequences=designed_seqs,
        struct_pred_model=struct_pred_model,
        tokenizer=tokenizer,
        return_aux=True,
    )
    aa_metrics_out = {f"aa_{k}": v for k, v in sc_metrics.items()}

    # Compute secondary structure content
    cropped_bb_coords = [c[..., [0, 1, 2, 4], :] for c in cropped_samp_coords]
    dssp_metrics = compute_secondary_structure_content(cropped_bb_coords)
    aa_metrics_out = {**aa_metrics_out, **dssp_metrics}

    # Compute bond length RMSE
    if two_stage_sampling:  # compute on original sample
        bond_rmse_coords = stage1_coords
    else:
        bond_rmse_coords = cropped_samp_coords
    bond_rmse = compute_bond_length_metric(
        bond_rmse_coords, cropped_samp_aatypes, samp_atom_mask
    )
    aa_metrics_out["aa_bond_rmse"] = bond_rmse

    # Convert codesign model back to allatom model and return metrics
    model.task = "allatom"
    model.remove_minimpnn()
    aa_aux_out = (
        cropped_samp_coords,
        cropped_samp_aatypes,
        samp_atom_mask,
        sc_aux["pred"],
        best_idx,
    )
    return aa_metrics_out, aa_aux_out
