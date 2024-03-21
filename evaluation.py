"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Utils for computing evaluation metrics and scaffolding benchmarks.
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
from tqdm import tqdm

from core import data
from core import residue_constants
from core import utils
from core import protein_mpnn as mpnn
import inference
import modules


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
        return _tmscore(aligned_coords1_ca, coords2[:, 1])
    elif metric == "tm_score_inv":
        tm = _tmscore(aligned_coords1_ca, coords2[:, 1])
        # Return 1 - tm score so sorts work properly
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
    sampling_config = inference.default_backbone_sampling_config()
    trimmed_coords, seq_mask = inference.draw_backbone_samples(
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

    sampling_config = inference.default_allatom_sampling_config()
    ret = inference.draw_allatom_samples(
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


# RFdiffusion supplement p84 (Watson et al)
RFDIFFUSION_SCAFFOLDING_BENCHMARKS = {
    "1PRW": {
        "contig_string": "5-20,A16-35,10-25,A52-71,5-20",
        "total_length": "60-105",
        "redesign_sequence": "A16-19,A21,A23,A25,A27-30,A32-35,A52-55,A57,A59,A61,A63-66,A68-71",
    },
    "1BCF": {
        "contig_string": "8-15,A92-99,16-30,A123-130,16-30,A47-54,16-30,A18-25,8-15",
        "total_length": "96-152",
        "redesign_sequence": "A19-25,A47-50,A52-53,A92-93,A95-99,A123-126,A128-129",
    },
    "1BCF_SITE": {
        "contig_string": "10-17,A94,25-40,A127-130,20-24,A51-54,16-30,A18,15-22",
        "total_length": "96-152",
        "redesign_sequence": "A52-53,A128-129",
    },
    "5TPN": {
        "contig_string": "10-40,A163-181,10-40",
        "total_length": "50-75",
        "redesign_sequence": "A163-168,A170-171,A179",
    },
    "5IUS": {
        "contig_string": "0-30,A119-140,15-40,A63-82,0-30",
        "total_length": "57-142",
        "redesign_sequence": "A63,A65,A67,A69,A71,A72,A76,A79,A80,A82,A119,A120,A121,A122,A123,A125,A127,A129,A130,A131,A133,A135,A137,A138,A140",
    },
    "3IXT": {
        "contig_string": "10-40,P254-277,10-40",
        "total_length": "50-75",
        "redesign_sequence": "P255,P258-259,P262-263,P268,P271-272,P275-276",
    },
    "5YUI": {
        "contig_string": "5-30,A93-97,5-20,A118-120,10-35,A198-200,10-30",
        "total_length": "50-100",
        "redesign_sequence": "A93,A95,A97,A118,A120",
    },
    "1QJG": {
        "contig_string": "10-20,A38,15-30,A14,15-30,A99,10-20",
        "total_length": "53-103",
        "redesign_sequence": "n/a",
    },
    "1QJG_NATIVE": {
        "contig_string": "10-20,A14,15-30,A38,50-70,A99,25-30",
        "total_length": "115-135",
        "redesign_sequence": "n/a",
    },
    "5AOU": {
        "contig_string": "40-60,A1051,20-40,A2083,20-35,A2110,100-140",
        "total_length": "230-270",
        "redesign_sequence": "n/a",
    },
    "5AOU_QUAD": {
        "contig_string": "40-60,A1051,20-40,A2083,20-35,A2110,60-80,A2180,40-60",
        "total_length": "230-270",
        "redesign_sequence": "n/a",
    },
    "7K4V": {
        "contig_string": "40-50,A44,3-8,A50,70-85,A127,150-200",
        "total_length": "280-320",
        "redesign_sequence": "n/a",
    },
    "1YCR": {
        "contig_string": "10-40,B19-27,10-40",
        "total_length": "40-100",
        "redesign_sequence": "B20-22,B24-25",
    },
    "2KL8": {
        "contig_string": "A1-7,20,A28-79",
        "total_length": "79",
        "redesign_sequence": "n/a",
    },
    "7MRX_60": {
        "contig_string": "0-38,B25-46,0-38",
        "total_length": "60",
        "redesign_sequence": "n/a",
    },
    "7MRX_85": {
        "contig_string": "0-63,B25-46,0-63",
        "total_length": "85",
        "redesign_sequence": "n/a",
    },
    "7MRX_128": {
        "contig_string": "0-122,B25-46,0-122",
        "total_length": "128",
        "redesign_sequence": "n/a",
    },
    "4JHW": {
        "contig_string": "10-25,F196-212,15-30,F63-69,10-25",
        "total_length": "60-90",
        "redesign_sequence": "F196,F198,F203,F211-212,F63,F69",
    },
    "4ZYP": {
        "contig_string": "10-40,A422-436,10-40",
        "total_length": "30-50",
        "redesign_sequence": "A422-427,A430-431,A433-436",
    },
    "5WN9": {
        "contig_string": "10-40,A170-189,10-40",
        "total_length": "35-50",
        "redesign_sequence": "A170-175,A188-189",
    },
    "6VW1": {
        "contig_string": "20-30,A24-42,4-10,A64-82,0-5",
        "total_length": "62-83",
        "redesign_sequence": "A25-26,A29-30,A32-34,A36-42,A64-82",
    },
    "5TRV_SHORT": {
        "contig_string": "0-35,A45-65,0-35",
        "total_length": "56",
        "redesign_sequence": "n/a",
    },
    "5TRV_MED": {
        "contig_string": "0-65,A45-65,0-65",
        "total_length": "86",
        "redesign_sequence": "n/a",
    },
    "5TRV_LONG": {
        "contig_string": "0-95,A45-65,0-95",
        "total_length": "116",
        "redesign_sequence": "n/a",
    },
    "6E6R_SHORT": {
        "contig_string": "0-35,A23-35,0-35",
        "total_length": "48",
        "redesign_sequence": "n/a",
    },
    "6E6R_MED": {
        "contig_string": "0-65,A23-35,0-65",
        "total_length": "78",
        "redesign_sequence": "n/a",
    },
    "6E6R_LONG": {
        "contig_string": "0-95,A23-35,0-95",
        "total_length": "108",
        "redesign_sequence": "n/a",
    },
    # Indices don't line up with PDB.
    # "6EXZ_SHORT": {
    #     "contig_string": "0-35,A28-42,0-35",
    #     "total_length": "50",
    #     "redesign_sequence": "n/a",
    # },
    # "6EXZ_MED": {
    #     "contig_string": "0-65,A28-42,0-65",
    #     "total_length": "80",
    #     "redesign_sequence": "n/a",
    # },
    # "6EXZ_LONG": {
    #     "contig_string": "0-95,A28-42,0-95",
    #     "total_length": "110",
    #     "redesign_sequence": "n/a",
    # },
}

SIDECHAIN_TIP_ATOMS = {
    "ALA": ["CA", "CB"],
    "ARG": ["CD", "CZ", "NE", "NH1", "NH2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "CYS": ["CA", "CB", "SG"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "GLN": ["CG", "CD", "NE2", "OE1"],
    "GLY": [],
    "HIS": ["CB", "CG", "CD2", "CE1", "ND1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CE", "NZ"],
    "MET": ["CG", "CE", "SD"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CA", "CB", "CG", "CD", "N"],
    "SER": ["CA", "CB", "OG"],
    "THR": ["CA", "CB", "CG2", "OG1"],
    "TRP": [
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "NE1",
    ],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}


def parse_span(span):
    if span[0].isalpha():  # conditional length
        chain_id = span[0]
        span = span[1:]
        if "-" in span:
            start, end = span.split("-")
            # this gives the zero-indexed indices
            list_of_aa_idxs = [
                (chain_id, idx) for idx in range(int(start), int(end) + 1)
            ]
        else:
            list_of_aa_idxs = [(chain_id, int(span))]
    else:  # generated length
        if "-" in span:
            start, end = span.split("-")
            # this gives the zero-indexed indices
            segment_length = torch.randint(int(start), int(end), (1,)).item()
            list_of_aa_idxs = [("", i) for i in range(segment_length)]
        else:
            list_of_aa_idxs = [("", i) for i in range(int(span))]

    return list_of_aa_idxs


def get_cond_seq_mask(span_in, redesign_sequence):
    # For a conditioning span, determine which residues are not to be redesigned.
    list_of_aa_idxs = parse_span(span_in)

    if redesign_sequence == "n/a":
        redesign_idxs = []
    else:
        redesign_idxs = []
        for span in "".join(redesign_sequence.upper().split()).split(","):
            redesign_idxs.extend(parse_span(span))

    span_cond_seq_mask = []
    for residue in list_of_aa_idxs:
        if residue in redesign_idxs:
            span_cond_seq_mask.append(0)
        else:
            span_cond_seq_mask.append(1)
    return torch.Tensor(span_cond_seq_mask)


def aatype_to_sidechain_end_mask(aatype):
    sidechain_end_mask = torch.zeros(aatype.shape[0], 37)
    for i, aa in enumerate(aatype):
        aa3 = residue_constants.restype_1to3[residue_constants.restypes[aa]]
        for atom_name in SIDECHAIN_TIP_ATOMS[aa3]:
            atom37_idx = residue_constants.atom_order[atom_name]
            sidechain_end_mask[i, atom37_idx] = 1
    return sidechain_end_mask.to(aatype.device)


def get_backbone_mask(atom_mask):
    backbone_mask = torch.zeros_like(atom_mask)
    for atom in ("N", "CA", "C", "O"):
        backbone_mask[:, residue_constants.atom_order[atom]] = 1
    return backbone_mask


def parse_scaffolding_task_to_sampling_inputs(
    task_name, seed=None, use_sidechain_end_atoms_only=False, device="cuda:0"
):
    if seed is not None:
        torch.manual_seed(seed)
    task_config = RFDIFFUSION_SCAFFOLDING_BENCHMARKS[task_name.upper()]
    contig_string = task_config["contig_string"]
    redesign_sequence = task_config["redesign_sequence"]
    cond_chain_id = set([c for c in task_config["contig_string"] if c.isalpha()]).pop()
    gt_feats = utils.load_feats_from_pdb(
        f"eval_pdbs/{task_name[:4].lower()}.pdb", chain_id=cond_chain_id
    )
    # gt_cond_atom_mask is 1s for all atoms, need to multiply by atom mask outside this func
    contig_string = "".join(contig_string.upper().split())
    seq_mask = []
    gt_cond_seq_mask = []
    gt_cond_atom_mask = []
    gt_aatype = []
    gt_coords = []
    current_length = 0
    for segment in contig_string.split(","):
        if segment[0].isalpha():
            parsed_span = parse_span(segment)
            cond_segment_idxs = [i for _, i in parsed_span]
            cond_segment_idxs = [
                (gt_feats["residue_index"] == i).nonzero().item()
                for i in cond_segment_idxs
            ]
            cond_len = len(parsed_span)
            seq_mask.append(torch.ones(cond_len))
            cond_seq_mask = get_cond_seq_mask(segment, redesign_sequence)
            keep_sidechain_mask = cond_seq_mask[:, None].tile((1, 37))
            backbone_mask = get_backbone_mask(keep_sidechain_mask)
            keep_sidechain_mask = keep_sidechain_mask.bool() | backbone_mask.bool()
            init_cond_atom_mask = gt_feats["atom_mask"][cond_segment_idxs]
            cond_atom_mask = (
                init_cond_atom_mask * keep_sidechain_mask
            )  # 1s for condseq mask = 1 and bb only for ==0
            cond_aatype = gt_feats["aatype"][cond_segment_idxs].long()
            if use_sidechain_end_atoms_only:
                sidechain_end_mask = aatype_to_sidechain_end_mask(cond_aatype)
                cond_atom_mask = cond_atom_mask * sidechain_end_mask
            gt_cond_seq_mask.append(cond_seq_mask)
            gt_cond_atom_mask.append(cond_atom_mask)
            gt_aatype.append((cond_aatype * cond_seq_mask).long())
            gt_coords.append(
                gt_feats["atom_positions"][cond_segment_idxs]
                * cond_atom_mask[..., None]
            )
            current_length += cond_len
        else:
            generated_len = len(parse_span(segment))
            seq_mask.append(torch.ones(generated_len))
            gt_cond_seq_mask.append(torch.zeros(generated_len))
            gt_cond_atom_mask.append(torch.zeros(generated_len, 37))
            gt_aatype.append(torch.zeros(generated_len).long())
            gt_coords.append(torch.zeros(generated_len, 37, 3))
            current_length += generated_len
    sampling_inputs = {
        "seq_mask": seq_mask,
        "gt_cond_seq_mask": gt_cond_seq_mask,
        "gt_cond_atom_mask": gt_cond_atom_mask,
        "gt_aatype": gt_aatype,
        "gt_coords": gt_coords,
    }
    sampling_inputs = {
        k: torch.cat(v)[None].to(device) for k, v in sampling_inputs.items()
    }
    sampling_inputs["gt_aatype"] = sampling_inputs["gt_aatype"].long()
    sampling_inputs["gt_coords"] = utils.center_coords_on_atom_mask(
        sampling_inputs["gt_coords"], sampling_inputs["gt_cond_atom_mask"]
    )
    return sampling_inputs


def batched_task_sampling_inputs(
    task_name, num_samples, seed=0, use_sidechain_end_atoms_only=False, device="cuda:0"
):
    all_inputs = [
        parse_scaffolding_task_to_sampling_inputs(
            task_name,
            seed + i,
            use_sidechain_end_atoms_only=use_sidechain_end_atoms_only,
        )
        for i in range(num_samples)
    ]
    longest_len = max([inputs["seq_mask"].shape[1] for inputs in all_inputs])
    batched_inputs = {"seq_mask": []}
    for inputs in all_inputs:
        for k, v in inputs.items():
            if k == "seq_mask":
                continue
            v, mask = data.make_fixed_size_1d(v[0].cpu(), longest_len)
            batched_inputs.setdefault(k, []).append(v)
        batched_inputs["seq_mask"].append(mask)
    batched_inputs = {k: torch.stack(v).to(device) for k, v in batched_inputs.items()}
    batched_inputs["gt_aatype"] = batched_inputs["gt_aatype"].long()
    return batched_inputs


def evaluate_scaffolding(
    model,
    seed=0,
    num_samples=10,
    use_sidechain_end_atoms_only=False,
    use_subset_of_tasks=False,
    struct_pred_model=None,
    tokenizer=None,
    save_dir="",
    verbose=False,
    **kwargs,
):
    # For each task, draw 10 samples, refold, and measure self-consistency
    # If scRMSD < 2 and motif_allatom_RMSD < 1.5 and pAE < 5 or pLDDT > 80, count as 'success'
    # Report success rate for each task
    if struct_pred_model is None:
        struct_pred_model, tokenizer = get_esmfold_model()
    sample_func = lambda **sample_kwargs: model.sample(
        return_last=False,
        return_aux=True,
        tqdm_pbar=tqdm,
        sidechain_mode=True,
        n_steps=500,
        **sample_kwargs,
    )
    if use_subset_of_tasks:
        subset_list = ["1BCF", "3IXT", "5IUS", "1QJG", "6VW1", "5TRV_SHORT"]
        benchmark = {
            k: v
            for k, v in RFDIFFUSION_SCAFFOLDING_BENCHMARKS.items()
            if k in subset_list
        }
    else:
        benchmark = RFDIFFUSION_SCAFFOLDING_BENCHMARKS
    if use_sidechain_end_atoms_only:
        benchmark = {
            k: v
            for k, v in benchmark.items()
            if v["redesign_sequence"] == "n/a" or k == "1BCF_SITE"
        }
    all_results = []
    for ti, (task_name, task_config) in enumerate(benchmark.items()):
        batch = batched_task_sampling_inputs(
            task_name,
            num_samples,
            use_sidechain_end_atoms_only=use_sidechain_end_atoms_only,
            seed=seed + ti,
        )
        aux = sample_func(
            **batch,
            **kwargs,
        )

        # Stage 2
        samp_seq = aux["st_traj"][-1]
        samp_coords = aux["xt_traj"][-1]
        seq_mask = aux["seq_mask"]
        cond_atom_mask = utils.atom37_mask_from_aatype((seq_mask * 7).long(), seq_mask)
        cond_atom_mask = cond_atom_mask.bool() | batch["gt_cond_atom_mask"].bool()
        stage_2_kwargs = vars(
            argparse.Namespace(
                apply_cond_proportion=1.0,
                n_steps=200,
                s_churn=100,
                step_scale=1.2,
                sidechain_mode=True,
                skip_mpnn_proportion=1.0,
            )
        )
        device = "cuda:0"
        stage2_aux = model.sample(
            gt_cond_atom_mask=cond_atom_mask.float().to(device),
            gt_cond_seq_mask=seq_mask.to(device),
            gt_coords=samp_coords.to(device),
            gt_aatype=samp_seq.to(device),
            seq_mask=seq_mask,
            return_last=False,
            return_aux=True,
            **stage_2_kwargs,
        )

        # Self consistency
        seq_lens = seq_mask.sum(-1).long()
        for i, l in enumerate(seq_lens):
            coords_i = stage2_aux["x"][i, :l]
            aatype_i = stage2_aux["s"][i, :l]
            pred_coords, plddts = predict_structures(
                utils.aatype_to_seq(aatype_i),
                model=struct_pred_model,
                tokenizer=tokenizer,
            )
            pred_coords_i = pred_coords[0]
            utils.write_coords_to_pdb(
                coords_i,
                os.path.join(
                    f"{save_dir}", f"{save_dir.split('/')[-2]}_{task_name}_samp_{i}.pdb"
                ),
                batched=False,
                aatype=aatype_i,
                atom_mask=utils.atom37_mask_from_aatype(aatype_i),
                conect=True,
            )
            utils.write_coords_to_pdb(
                pred_coords_i,
                os.path.join(
                    f"{save_dir}", f"{save_dir.split('/')[-2]}_{task_name}_pred_{i}.pdb"
                ),
                batched=False,
                aatype=aatype_i,
                atom_mask=utils.atom37_mask_from_aatype(aatype_i),
                conect=True,
            )
            motif_atom_mask = batch["gt_cond_atom_mask"][i, :l]
            motif_bb_atom_mask = motif_atom_mask * get_backbone_mask(motif_atom_mask)
            alignment = utils.quick_tmalign(coords_i, coords_i, pred_coords_i)

            def masked_rmsd(aligned, coords2, atom_mask):
                rmsd = (aligned - coords2).pow(2).sum(-1).sqrt()
                rmsd = (rmsd * atom_mask).sum() / atom_mask.sum().clamp(min=1)
                return rmsd.cpu().item()

            # Compute motif RMSDs on TMalignment, scRMSD on kabsch alignment
            motif_rmsd = masked_rmsd(
                alignment["aligned"], pred_coords_i, motif_atom_mask
            )
            motif_rmsd = compute_masked_rmsd(coords_i, pred_coords_i, motif_atom_mask)
            motif_bb_rmsd = masked_rmsd(
                alignment["aligned"], pred_coords_i, motif_bb_atom_mask
            )
            motif_bb_rmsd = compute_masked_rmsd(
                coords_i, pred_coords_i, motif_bb_atom_mask
            )
            sc_rmsd = compute_structure_metric(coords_i, pred_coords_i)
            sc_tm = alignment["tm_score"]
            result = {
                "task_name": task_name,
                "sample_idx": i,
                "motif_rmsd": motif_rmsd,
                "motif_bb_rmsd": motif_bb_rmsd,
                "sc_rmsd": sc_rmsd,
                "sc_tm": sc_tm,
                "plddt": plddts[0].cpu().item(),
                "motif_idxs": motif_atom_mask.any(-1).nonzero().squeeze(-1).tolist(),
            }
            if verbose:
                print(result)
            all_results.append(result)

    return all_results
