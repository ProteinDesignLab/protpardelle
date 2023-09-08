"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Various utils for handling protein data.
"""

import os
import shlex
import subprocess
import sys
import torch
import yaml
import argparse

from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.functional as F
import Bio
from Bio.PDB.DSSP import DSSP

from core import protein
from core import protein_mpnn
from core import residue_constants


PATH_TO_TMALIGN = "/home/alexechu/essentials_kit/ml_utils/align/TMalign/TMalign"


################ STRUCTURE/FORMAT UTILS #############################


def aatype_to_seq(aatype, seq_mask=None):
    if seq_mask is None:
        seq_mask = torch.ones_like(aatype)

    mapping = residue_constants.restypes_with_x
    mapping = mapping + ["<mask>"]

    unbatched = False
    if len(aatype.shape) == 1:
        unbatched = True
        aatype = [aatype]
        seq_mask = [seq_mask]

    seqs = []
    for i, ai in enumerate(aatype):
        seq = []
        for j, aa in enumerate(ai):
            if seq_mask[i][j] == 1:
                try:
                    seq.append(mapping[aa])
                except IndexError:
                    print(aatype[i])
                    raise Exception(f"Error in mapping {aa} at {i},{j}")
        seqs.append("".join(seq))

    if unbatched:
        seqs = seqs[0]
    return seqs


def seq_to_aatype(seq, num_tokens=21):
    if num_tokens == 20:
        mapping = residue_constants.restype_order
    if num_tokens == 21:
        mapping = residue_constants.restype_order_with_x
    if num_tokens == 22:
        mapping = residue_constants.restype_order_with_x
        mapping["<mask>"] = 21
    return torch.Tensor([mapping[aa] for aa in seq]).long()


def batched_seq_to_aatype_and_mask(seqs, max_len=None):
    if max_len is None:
        max_len = max([len(s) for s in seqs])
    aatypes = []
    seq_mask = []
    for s in seqs:
        pad_size = max_len - len(s)
        aatype = seq_to_aatype(s)
        aatypes.append(F.pad(aatype, (0, pad_size)))
        mask = torch.ones_like(aatype).float()
        seq_mask.append(F.pad(mask, (0, pad_size)))
    return torch.stack(aatypes), torch.stack(seq_mask)


def atom37_mask_from_aatype(aatype, seq_mask=None):
    # source_mask is (21,37) originally
    source_mask = torch.Tensor(residue_constants.restype_atom37_mask).to(aatype.device)
    bb_atoms = source_mask[residue_constants.restype_order["G"]][None]
    # Use only the first 20 plus bb atoms for X, mask
    source_mask = torch.cat([source_mask[:-1], bb_atoms, bb_atoms], 0)
    atom_mask = source_mask[aatype]
    if seq_mask is not None:
        atom_mask *= seq_mask[..., None]
    return atom_mask


def atom37_coords_from_atom14(atom14_coords, aatype, return_mask=False):
    # Unbatched
    device = atom14_coords.device
    atom37_coords = torch.zeros((atom14_coords.shape[0], 37, 3)).to(device)
    for i in range(atom14_coords.shape[0]):  # per residue
        aa = aatype[i]
        aa_3name = residue_constants.restype_1to3[residue_constants.restypes[aa]]
        atom14_atoms = residue_constants.restype_name_to_atom14_names[aa_3name]
        for j in range(14):
            atom_name = atom14_atoms[j]
            if atom_name != "":
                atom37_idx = residue_constants.atom_order[atom_name]
                atom37_coords[i, atom37_idx, :] = atom14_coords[i, j, :]

    if return_mask:
        atom37_mask = atom37_mask_from_aatype(aatype)
        return atom37_coords, atom37_mask
    return atom37_coords


def atom73_mask_from_aatype(aatype, seq_mask=None):
    source_mask = torch.Tensor(residue_constants.restype_atom73_mask).to(aatype.device)
    atom_mask = source_mask[aatype]
    if seq_mask is not None:
        atom_mask *= seq_mask[..., None]
    return atom_mask


def atom37_to_atom73(atom37, aatype, return_mask=False):
    # Unbatched
    atom73 = torch.zeros((atom37.shape[0], 73, 3)).to(atom37)
    for i in range(atom37.shape[0]):
        aa = aatype[i]
        aa1 = residue_constants.restypes[aa]
        for j, atom37_name in enumerate(residue_constants.atom_types):
            atom73_name = atom37_name
            if atom37_name not in ["N", "CA", "C", "O", "CB"]:
                atom73_name = aa1 + atom73_name
            if atom73_name in residue_constants.atom73_names_to_idx:
                atom73_idx = residue_constants.atom73_names_to_idx[atom73_name]
                atom73[i, atom73_idx, :] = atom37[i, j, :]

    if return_mask:
        atom73_mask = atom73_mask_from_aatype(aatype)
        return atom73, atom73_mask
    return atom73


def atom73_to_atom37(atom73, aatype, return_mask=False):
    # Unbatched
    atom37_coords = torch.zeros((atom73.shape[0], 37, 3)).to(atom73)
    for i in range(atom73.shape[0]):  # per residue
        aa = aatype[i]
        aa1 = residue_constants.restypes[aa]
        for j, atom_type in enumerate(residue_constants.atom_types):
            atom73_name = atom_type
            if atom73_name not in ["N", "CA", "C", "O", "CB"]:
                atom73_name = aa1 + atom73_name
            if atom73_name in residue_constants.atom73_names_to_idx:
                atom73_idx = residue_constants.atom73_names_to_idx[atom73_name]
                atom37_coords[i, j, :] = atom73[i, atom73_idx, :]

    if return_mask:
        atom37_mask = atom37_mask_from_aatype(aatype)
        return atom37_coords, atom37_mask
    return atom37_coords


def get_dmap(pdb, atoms=["N", "CA", "C", "O"], batched=True, out="torch", device=None):
    def _dmap_from_coords(coords):
        coords = coords.contiguous()
        dmaps = torch.cdist(coords, coords).unsqueeze(1)
        if out == "numpy":
            return dmaps.detach().cpu().numpy()
        elif out == "torch":
            if device is not None:
                return dmaps.to(device)
            else:
                return dmaps

    if isinstance(pdb, str):  # input is pdb file
        coords = load_coords_from_pdb(pdb, atoms=atoms).view(1, -1, 3)
        return _dmap_from_coords(coords)
    elif len(pdb.shape) == 2:  # single set of coords
        if isinstance(pdb, np.ndarray):
            pdb = torch.Tensor(pdb)
        return _dmap_from_coords(pdb.unsqueeze(0))
    elif len(pdb.shape) == 3 and batched:
        return _dmap_from_coords(pdb)
    elif len(pdb.shape) == 3 and not batched:
        return _dmap_from_coords(pdb.view(1, -1, 3))
    elif len(pdb.shape) == 4:
        return _dmap_from_coords(pdb.view(pdb.size(0), -1, 3))


def get_channeled_dmap(coords):
    # coords is b, nres, natom, 3
    coords = coords.permute(0, 2, 1, 3)
    dvecs = coords[..., None, :] - coords[..., None, :, :]  # b, natom, nres, nres, 3
    dists = torch.sqrt(dvecs.pow(2).sum(-1) + 1e-8)
    return dists


def fill_in_cbeta_for_atom37(coords):
    b = coords[..., 1, :] - coords[..., 0, :]
    c = coords[..., 2, :] - coords[..., 1, :]
    a = torch.cross(b, c, dim=-1)
    cbeta = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + coords[..., 1, :]
    new_coords = torch.clone(coords)
    new_coords[..., 3, :] = cbeta
    return new_coords


def get_distogram(coords, n_bins=20, start=2, return_onehot=True, seq_mask=None):
    # coords is b, nres, natom, 3
    # distogram for cb atom (assume 3rd atom)
    coords_with_cb = fill_in_cbeta_for_atom37(coords)
    dists = get_channeled_dmap(coords_with_cb[:, :, 3:4]).squeeze(1)
    bins = torch.arange(start, start + n_bins - 1).to(dists.device)
    dgram = torch.bucketize(dists, bins)
    dgram_oh = F.one_hot(dgram, n_bins)
    if seq_mask is not None:
        mask_2d = seq_mask[:, :, None] * seq_mask[:, None, :]
        dgram = dgram * mask_2d
        dgram_oh = dgram_oh * mask_2d[..., None]

    if return_onehot:
        return dgram_oh
    return dgram


def get_contacts(coords=None, distogram=None, seq_mask=None):
    if distogram is None:
        distogram = get_distogram(coords)
    contacts = (distogram.argmax(-1) < 6).float()
    if seq_mask is not None:
        contacts *= seq_mask[..., None] * seq_mask[..., None, :]
    return contacts


def dihedral(a, b, c, d):
    # inputs can be (1,3), (n,3), or (bs,n,3)
    b1 = a - b
    b2 = b - c
    b3 = c - d
    n1 = F.normalize(torch.cross(b1, b2), dim=-1)
    n2 = F.normalize(torch.cross(b2, b3), dim=-1)
    m1 = torch.cross(n1, b2 / b2.norm(dim=-1).unsqueeze(-1))
    y = (m1 * n2).sum(dim=-1)
    x = (n1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def get_torsions_from_coords(
    coords, atoms=["N", "CA", "C", "O"], batched=True, out="torch", device=None
):
    """
    Returns a n-dim array of shape (bs, nres, ntors), where ntors is the
    number of torsion angles (e.g. 2 if using phi and psi), with units of radians.
    """
    if isinstance(coords, np.ndarray):
        coords = torch.Tensor(coords)
    if len(coords.shape) == 2:
        coords = coords.unsqueeze(0)
    if len(coords.shape) == 4:
        coords = coords.view(coords.size(0), -1, 3)
    if len(coords.shape) == 3 and not batched:
        coords = coords.view(1, -1, 3)
    if len(coords.shape) == 3:
        bs = coords.size(0)
        if "O" in atoms:
            idxs = [
                i for i in range(coords.size(1)) if i % 4 != 3
            ]  # deselect O atoms for N-Ca-C-O coords
            coords = coords[:, idxs, :]
        a, b, c, d = (
            coords[:, :-3, :],
            coords[:, 1:-2, :],
            coords[:, 2:-1, :],
            coords[:, 3:, :],
        )
        torsions = dihedral(
            a, b, c, d
        )  # output order is psi-omega-phi, reorganize to (bs, nres, 3)
        torsions = torsions.view(bs, torsions.size(1) // 3, 3)
        omegaphi = torch.cat(
            (torch.zeros(bs, 1, 2).to(coords.device), torsions[:, :, 1:]), 1
        )
        psi = torch.cat((torsions[:, :, 0], torch.zeros(bs, 1).to(coords.device)), 1)
        torsions = torch.cat(
            (
                omegaphi[:, :, 1].unsqueeze(-1),
                psi.unsqueeze(-1),
                omegaphi[:, :, 0].unsqueeze(-1),
            ),
            -1,
        )
    else:
        raise Exception("input coords not of correct dims")

    if out == "numpy":
        return torsions.detach().cpu().numpy()
    elif out == "torch":
        if device is not None:
            return torsions.to(device)
        else:
            return torsions


def get_trig_from_torsions(torsions, out="torch", device=None):
    """
    Calculate unit circle projections from coords input.

    Returns a n-dim array of shape (bs, nres, ntors, 2), where ntors is the
    number of torsion angles (e.g. 2 if using phi and psi), and the last
    dimension is the xy unit-circle coordinates of the corresponding angle.
    """
    if isinstance(torsions, np.ndarray):
        torsions = torch.Tensor(torsions)
    x = torsions.cos()
    y = torsions.sin()
    trig = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), -1)
    if out == "numpy":
        return trig.detach().cpu().numpy()
    elif out == "torch":
        if device is not None:
            return trig.to(device)
        else:
            return trig


def get_abego_string_from_torsions(torsions):
    A_bin = (-75, 50)
    G_bin = (-100, 100)
    torsions = torsions * 180.0 / np.pi
    phi, psi = torsions[:, :, 0], torsions[:, :, 1]
    abego_vec = np.zeros((torsions.size(0), torsions.size(1))).astype(str)
    A = (phi <= 0) & (psi <= A_bin[1]) & (psi > A_bin[0])
    B = (phi <= 0) & ((psi > A_bin[1]) | (psi <= A_bin[0]))
    G = (phi > 0) & (psi <= G_bin[1]) & (psi > G_bin[0])
    E = (phi > 0) & ((psi > G_bin[1]) | (psi <= G_bin[0]))
    abego_vec[A] = "A"
    abego_vec[B] = "B"
    abego_vec[G] = "G"
    abego_vec[E] = "E"
    abego_strs = ["".join(v) for v in abego_vec]
    return abego_strs


def get_bond_lengths_from_coords(coords, batched=True, out="torch", device=None):
    """
    Returns array of shape (bs, n_res, 4), where final dim is bond lengths
    in order of N-Ca, Ca-C, C-O, C-N (none for last residue)
    """
    if isinstance(coords, np.ndarray):
        coords = torch.Tensor(coords)
    if len(coords.shape) == 2:
        coords = coords.unsqueeze(0)
    if len(coords.shape) == 3 and not batched:
        coords = coords.view(1, -1, 3)
    if len(coords.shape) == 4:
        coords = coords.view(coords.size(0), -1, 3)
    N = coords[:, ::4, :]
    Ca = coords[:, 1::4, :]
    C = coords[:, 2::4, :]
    O = coords[:, 3::4, :]
    NCa = (Ca - N).norm(dim=-1).unsqueeze(-1)
    CaC = (C - Ca).norm(dim=-1).unsqueeze(-1)
    CO = (O - C).norm(dim=-1).unsqueeze(-1)
    CN = (N[:, 1:] - C[:, :-1]).norm(dim=-1)
    CN = torch.cat([CN, torch.zeros(CN.size(0), 1).to(CN.device)], 1).unsqueeze(-1)
    blengths = torch.cat((NCa, CaC, CO, CN), -1)
    if out == "numpy":
        return blengths.detach().cpu().numpy()
    elif out == "torch":
        if device is not None:
            return blengths.to(device)
        else:
            return blengths


def get_bond_angles_from_coords(coords, batched=True, out="torch", device=None):
    """
    Returns array of shape (bs, n_res, 5), where final dim is bond angles
    in order of N-Ca-C, Ca-C-O, Ca-C-N, O-C-N, C-N-Ca (none for last residue)
    """

    def _angle(v1, v2):
        cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1))
        return cos.acos()

    if isinstance(coords, np.ndarray):
        coords = torch.Tensor(coords)
    if len(coords.shape) == 2:
        coords = coords.unsqueeze(0)
    if len(coords.shape) == 3 and not batched:
        coords = coords.view(1, -1, 3)
    if len(coords.shape) == 4:
        coords = coords.view(coords.size(0), -1, 3)
    N = coords[:, ::4, :]
    Nnext = coords[:, 4::4, :]
    Ca = coords[:, 1::4, :]
    Canext = coords[:, 5::4, :]
    C = coords[:, 2::4, :]
    O = coords[:, 3::4, :]
    CaN = N - Ca
    CaC = C - Ca
    CCa = Ca - C
    CO = O - C
    CNnext = Nnext - C[:, :-1, :]
    NnextC = -1 * CNnext
    NnextCanext = Canext - Nnext
    NCaC = _angle(CaN, CaC).unsqueeze(-1)
    CaCO = _angle(CCa, CO).unsqueeze(-1)
    CaCN = _angle(CCa[:, :-1], CNnext).unsqueeze(-1)
    CaCN = _extend(CaCN)
    OCN = _angle(CO[:, :-1], CNnext).unsqueeze(-1)
    OCN = _extend(OCN)
    CNCa = _angle(NnextC, NnextCanext).unsqueeze(-1)
    # CNCa = torch.cat([CNCa, torch.zeros(CNCa.size(0), 1).to(CNCa.device)], 1).unsqueeze(-1)
    CNCa = _extend(CNCa)
    bangles = torch.cat((NCaC, CaCO, CaCN, OCN, CNCa), -1)
    if out == "numpy":
        return bangles.detach().cpu().numpy()
    elif out == "torch":
        if device is not None:
            return bangles.to(device)
        else:
            return bangles


def get_buried_positions_mask(coords, seq_mask=None, threshold=6.0):
    ca_idx = residue_constants.atom_order["CA"]  # typically 1
    cb_idx = residue_constants.atom_order["CB"]  # typically 3
    if seq_mask is None:
        seq_mask = torch.ones_like(coords)[..., 0, 0]
    coords = fill_in_cbeta_for_atom37(coords)

    # get 8 closest neighbors by CB
    neighbor_coords = coords[:, :, cb_idx]

    ca_neighbor_dists, edge_index = protein_mpnn.get_closest_neighbors(
        neighbor_coords, seq_mask, 9
    )
    edge_index = edge_index[..., 1:].contiguous()

    # compute avg CB distance
    cb_coords = coords[:, :, cb_idx]
    neighbor_cb = protein_mpnn.gather_nodes(cb_coords, edge_index)
    avg_cb_dist = (neighbor_cb - cb_coords[..., None, :]).pow(2).sum(-1).sqrt().mean(-1)

    buried_positions_mask = (avg_cb_dist < threshold).float() * seq_mask
    return buried_positions_mask


def get_fullatom_bond_lengths_from_coords(
    coords, aatype, atom_mask=None, return_format="per_aa"
):
    # Also return sidechain bond angles. All unbatched. return list of dicts
    def dist(xyz1, xyz2):
        return (xyz1 - xyz2).pow(2).sum().sqrt().detach().cpu().item()

    assert aatype.max() <= 19
    seq = aatype_to_seq(aatype)
    # residue-wise list of dicts [{'N-CA': a, 'CA-C': b}, {'N-CA': a, 'CA-C': b}]
    all_bond_lens_by_pos = []
    # aa-wise dict of dicts of lists {'A': {'N-CA': [a, b, c], 'CA-C': [a, b, c]}}
    all_bond_lens_by_aa = {aa: {} for aa in residue_constants.restypes}
    for i, res in enumerate(coords):
        aa3 = residue_constants.restype_1to3[seq[i]]
        res_bond_lens = {}
        for j, atom1 in enumerate(residue_constants.atom_types):
            for k, atom2 in enumerate(residue_constants.atom_types):
                if j < k and protein.are_atoms_bonded(aa3, atom1, atom2):
                    if atom_mask is None or (
                        atom_mask[i, j] > 0.5 and atom_mask[i, k] > 0.5
                    ):
                        bond_name = f"{atom1}-{atom2}"
                        bond_len = dist(res[j], res[k])
                        res_bond_lens[bond_name] = bond_len
        all_bond_lens_by_pos.append(res_bond_lens)
        for key, val in res_bond_lens.items():
            all_bond_lens_by_aa[seq[i]].setdefault(key, []).append(val)

    if return_format == "per_aa":
        return all_bond_lens_by_aa
    elif return_format == "per_position":
        return all_bond_lens_by_pos


def batched_fullatom_bond_lengths_from_coords(
    coords, aatype, atom_mask=None, return_format="per_aa"
):
    # Expects trimmed coords (no mask)
    if return_format == "per_position":
        batched_bond_lens = []
    elif return_format == "per_aa":
        batched_bond_lens = {aa: {} for aa in residue_constants.restypes}
    for i, c in enumerate(coords):
        atom_mask_i = None if atom_mask is None else atom_mask[i]
        bond_lens = get_fullatom_bond_lengths_from_coords(
            c, aatype[i], atom_mask=atom_mask_i, return_format=return_format
        )
        if return_format == "per_position":
            batched_bond_lens.extend(bond_lens)
        elif return_format == "per_aa":
            for aa, d in bond_lens.items():
                for bond, lengths in d.items():
                    batched_bond_lens[aa].setdefault(bond, []).extend(lengths)
    return batched_bond_lens


def batched_fullatom_bond_angles_from_coords(coords, aatype, return_format="per_aa"):
    # Expects trimmed coords (no mask)
    if return_format == "per_position":
        batched_bond_angles = []
    elif return_format == "per_aa":
        batched_bond_angles = {aa: {} for aa in residue_constants.restypes}
    for i, c in enumerate(coords):
        bond_angles = get_fullatom_bond_angles_from_coords(
            c, aatype[i], return_format=return_format
        )
        if return_format == "per_position":
            batched_bond_angles.extend(bond_angles)
        elif return_format == "per_aa":
            for aa, d in bond_angles.items():
                for bond, lengths in d.items():
                    batched_bond_angles[aa].setdefault(bond, []).extend(lengths)
    return batched_bond_angles


def get_chi_angles(coords, aatype, atom_mask=None, seq_mask=None):
    # unbatched
    # return (n, 4) chis in degrees and mask
    chis = []
    chi_mask = []
    atom_order = residue_constants.atom_order

    seq = aatype_to_seq(aatype, seq_mask=seq_mask)

    for i, aa1 in enumerate(seq):  # per residue
        if seq_mask is not None and seq_mask[i] == 0:
            chis.append([0, 0, 0, 0])
            chi_mask.append([0, 0, 0, 0])
        else:
            chi = []
            mask = []
            chi_atoms = residue_constants.chi_angles_atoms[
                residue_constants.restype_1to3[aa1]
            ]
            for j in range(4):  # per chi angle
                if j > len(chi_atoms) - 1:
                    chi.append(0)
                    mask.append(0)
                elif atom_mask is not None and any(
                    [atom_mask[i, atom_order[a]] < 0.5 for a in chi_atoms[j]]
                ):
                    chi.append(0)
                    mask.append(0)
                else:
                    # Four atoms per dihedral
                    xyz4 = [coords[i, atom_order[a]] for a in chi_atoms[j]]
                    angle = dihedral(*xyz4) * 180 / np.pi
                    chi.append(angle)
                    mask.append(1)
            chis.append(chi)
            chi_mask.append(mask)

    chis = torch.Tensor(chis)
    chi_mask = torch.Tensor(chi_mask)

    return chis, chi_mask


def fill_Os_from_NCaC_coords(
    coords: torch.Tensor, out: str = "torch", device: str = None
):
    """Given NCaC coords, add O atom coordinates in.
    (bs, 3n, 3) -> (bs, 4n, 3)
    """
    CO_LEN = 1.231
    if len(coords.shape) == 2:
        coords = coords.unsqueeze(0)
    Cs = coords[:, 2:-1:3, :]  # all but last C
    CCa_norm = F.normalize(coords[:, 1:-2:3, :] - Cs, dim=-1)  # all but last Ca
    CN_norm = F.normalize(coords[:, 3::3, :] - Cs, dim=-1)  # all but first N
    Os = F.normalize(CCa_norm + CN_norm, dim=-1) * -CO_LEN
    Os += Cs
    # TODO place C-term O atom properly
    Os = torch.cat([Os, coords[:, -1, :].view(-1, 1, 3) + 1], 1)
    coords_out = []
    for i in range(Os.size(1)):
        coords_out.append(coords[:, i * 3 : (i + 1) * 3, :])
        coords_out.append(Os[:, i, :].view(-1, 1, 3))
    coords_out = torch.cat(coords_out, 1)
    if out == "numpy":
        return coords_out.detach().cpu().numpy()
    elif out == "torch":
        if device is not None:
            return coords_out.to(device)
        else:
            return coords_out


def _extend(x, axis=1, n=1, prepend=False):
    # Add an extra zeros 'residue' to the end (or beginning, prepend=True) of a Tensor
    # Used to extend torsions when there is no 'psi' for last residue
    shape = list(x.shape)
    shape[axis] = n
    if prepend:
        return torch.cat([torch.zeros(shape).to(x.device), x], axis)
    else:
        return torch.cat([x, torch.zeros(shape).to(x.device)], axis)


def trim_coords(coords, n_res, batched=True):
    if batched:  # Return list of tensors
        front = (coords.shape[1] - n_res) // 2
        return [
            coords[i, front[i] : front[i] + n_res[i]] for i in range(coords.shape[0])
        ]
    else:
        if isinstance(n_res, torch.Tensor):
            n_res = n_res.int()
        front_pad = (coords.shape[0] - n_res) // 2
        return coords[front_pad : front_pad + n_res]


def batch_align_on_calpha(x, y):
    aligned_x = []
    for i, xi in enumerate(x):
        xi_calpha = xi[:, 1, :]
        _, (R, t) = kabsch_align(xi_calpha, y[i, :, 1, :])
        xi_ctr = xi - xi_calpha.mean(0, keepdim=True)
        xi_aligned = xi_ctr @ R.t() + t
        aligned_x.append(xi_aligned)
    return torch.stack(aligned_x)


def kabsch_align(p, q):
    if len(p.shape) > 2:
        p = p.reshape(-1, 3)
    if len(q.shape) > 2:
        q = q.reshape(-1, 3)
    p_ctr = p - p.mean(0, keepdim=True)
    t = q.mean(0, keepdim=True)
    q_ctr = q - t
    H = p_ctr.t() @ q_ctr
    U, S, V = torch.svd(H)
    R = V @ U.t()
    I_ = torch.eye(3).to(p)
    I_[-1, -1] = R.det().sign()
    R = V @ I_ @ U.t()
    p_aligned = p_ctr @ R.t() + t
    return p_aligned, (R, t)


def get_dssp_string(pdb):
    try:
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb[:-3], pdb)
        dssp = DSSP(structure[0], pdb, dssp="mkdssp")
        dssp_string = "".join([dssp[k][2] for k in dssp.keys()])
        return dssp_string
    except Exception as e:
        print(e)
        return None


def pool_dssp_symbols(dssp_string, newchar=None, chars=["-", "T", "S", "C", " "]):
    """Replaces all instances of chars with newchar. DSSP chars are helix=GHI, strand=EB, loop=- TSC"""
    if newchar is None:
        newchar = chars[0]
    string_out = dssp_string
    for c in chars:
        string_out = string_out.replace(c, newchar)
    return string_out


def get_3state_dssp(pdb=None, coords=None):
    if coords is not None:
        pdb = "temp_dssp.pdb"
        write_coords_to_pdb(coords, pdb, batched=False)
    dssp_string = get_dssp_string(pdb)
    if dssp_string is not None:
        dssp_string = pool_dssp_symbols(dssp_string, newchar="L")
        dssp_string = pool_dssp_symbols(dssp_string, chars=["H", "G", "I"])
        dssp_string = pool_dssp_symbols(dssp_string, chars=["E", "B"])
    if coords is not None:
        subprocess.run(shlex.split(f"rm {pdb}"))
    return dssp_string


############## SAVE/LOAD UTILS #################################


def load_feats_from_pdb(
    pdb, bb_atoms=["N", "CA", "C", "O"], load_atom73=False, **kwargs
):
    feats = {}
    with open(pdb, "r") as f:
        pdb_str = f.read()
    protein_obj = protein.from_pdb_string(pdb_str, **kwargs)
    bb_idxs = [residue_constants.atom_order[a] for a in bb_atoms]
    bb_coords = torch.from_numpy(protein_obj.atom_positions[:, bb_idxs])
    feats["bb_coords"] = bb_coords.float()
    for k, v in vars(protein_obj).items():
        feats[k] = torch.Tensor(v)
    feats["aatype"] = feats["aatype"].long()
    if load_atom73:
        feats["atom73_coords"], feats["atom73_mask"] = atom37_to_atom73(
            feats["atom_positions"], feats["aatype"], return_mask=True
        )
    return feats


def load_coords_from_pdb(
    pdb,
    atoms=["N", "CA", "C", "O"],
    method="raw",
    also_bfactors=False,
    normalize_bfactors=True,
):
    """Returns array of shape (1, n_res, len(atoms), 3)"""
    coords = []
    bfactors = []
    if method == "raw":  # Raw numpy implementation, faster than biopdb
        # Indexing into PDB format, allowing XXXX.XXX
        coords_in_pdb = [slice(30, 38), slice(38, 46), slice(46, 54)]
        # Indexing into PDB format, allowing XXX.XX
        bfactor_in_pdb = slice(60, 66)

        with open(pdb, "r") as f:
            resi_prev = 1
            counter = 0
            for l in f:
                l_split = l.rstrip("\n").split()
                if len(l_split) > 0 and l_split[0] == "ATOM" and l_split[2] in atoms:
                    resi = l_split[5]
                    if resi == resi_prev:
                        counter += 1
                    else:
                        counter = 0
                    if counter < len(atoms):
                        xyz = [
                            np.array(l[s].strip()).astype(float) for s in coords_in_pdb
                        ]
                        coords.append(xyz)
                        if also_bfactors:
                            bfactor = np.array(l[bfactor_in_pdb].strip()).astype(float)
                            bfactors.append(bfactor)
                    resi_prev = resi
            coords = torch.Tensor(np.array(coords)).view(1, -1, len(atoms), 3)
            if also_bfactors:
                bfactors = torch.Tensor(np.array(bfactors)).view(1, -1, len(atoms))
    elif method == "biopdb":
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb[:-3], pdb)
        for model in structure:
            for chain in model:
                for res in chain:
                    for atom in atoms:
                        try:
                            coords.append(np.asarray(res[atom].get_coord()))
                            if also_bfactors:
                                bfactors.append(np.asarray(res[atom].get_bfactor()))
                        except:
                            continue
    else:
        raise NotImplementedError(f"Invalid method for reading coords: {method}")
    if also_bfactors:
        if normalize_bfactors:  # Normalize over Calphas
            mean_b = bfactors[..., 1].mean()
            std_b = bfactors[..., 1].var().sqrt()
            bfactors = (bfactors - mean_b) / (std_b + 1e-6)
        return coords, bfactors
    return coords


def feats_to_pdb_str(
    atom_positions,
    aatype=None,
    atom_mask=None,
    residue_index=None,
    chain_index=None,
    b_factors=None,
    atom_lines_only=True,
    conect=False,
    **kwargs,
):
    # Expects unbatched, cropped inputs. needs at least one of atom_mask, aatype
    # Uses all-GLY aatype if aatype not given: does not infer from atom_mask
    assert aatype is not None or atom_mask is not None
    if atom_mask is None:
        aatype = aatype.cpu()
        atom_mask = atom37_mask_from_aatype(aatype, torch.ones_like(aatype))
    if aatype is None:
        seq_mask = atom_mask[:, residue_constants.atom_order["CA"]].cpu()
        aatype = seq_mask * residue_constants.restype_order["G"]
    if residue_index is None:
        residue_index = torch.arange(aatype.shape[-1])
    if chain_index is None:
        chain_index = torch.ones_like(aatype)
    if b_factors is None:
        b_factors = torch.ones_like(atom_mask)

    cast = lambda x: np.array(x.detach().cpu()) if isinstance(x, torch.Tensor) else x
    prot = protein.Protein(
        atom_positions=cast(atom_positions),
        atom_mask=cast(atom_mask),
        aatype=cast(aatype),
        residue_index=cast(residue_index),
        chain_index=cast(chain_index),
        b_factors=cast(b_factors),
    )
    pdb_str = protein.to_pdb(prot, conect=conect)
    if conect:
        pdb_str, conect_str = pdb_str
    if atom_lines_only:
        pdb_lines = pdb_str.split("\n")
        atom_lines = [
            l for l in pdb_lines if len(l.split()) > 1 and l.split()[0] == "ATOM"
        ]
        pdb_str = "\n".join(atom_lines) + "\n"
    if conect:
        pdb_str = pdb_str + conect_str
    return pdb_str


def bb_coords_to_pdb_str(coords, atoms=["N", "CA", "C", "O"]):
    def _bb_pdb_line(atom, atomnum, resnum, coords, elem, res="GLY"):
        atm = "ATOM".ljust(6)
        atomnum = str(atomnum).rjust(5)
        atomname = atom.center(4)
        resname = res.ljust(3)
        chain = "A".rjust(1)
        resnum = str(resnum).rjust(4)
        x = str("%8.3f" % (float(coords[0]))).rjust(8)
        y = str("%8.3f" % (float(coords[1]))).rjust(8)
        z = str("%8.3f" % (float(coords[2]))).rjust(8)
        occ = str("%6.2f" % (float(1))).rjust(6)
        temp = str("%6.2f" % (float(20))).ljust(6)
        elname = elem.rjust(12)
        return "%s%s %s %s %s%s    %s%s%s%s%s%s\n" % (
            atm,
            atomnum,
            atomname,
            resname,
            chain,
            resnum,
            x,
            y,
            z,
            occ,
            temp,
            elname,
        )

    n = coords.shape[0]
    na = len(atoms)
    pdb_str = ""
    for j in range(0, n, na):
        for idx, atom in enumerate(atoms):
            pdb_str += _bb_pdb_line(
                atom,
                j + idx + 1,
                (j + na) // na,
                coords[j + idx],
                atom[0],
            )
    return pdb_str


def write_coords_to_pdb(
    coords_in,
    filename,
    batched=True,
    write_to_frames=False,
    conect=False,
    **all_atom_feats,
):
    def _write_pdb_string(pdb_str, filename, append=False):
        write_mode = "a" if append else "w"
        with open(filename, write_mode) as f:
            if write_to_frames:
                f.write("MODEL\n")
            f.write(pdb_str)
            if write_to_frames:
                f.write("ENDMDL\n")

    if not (batched or write_to_frames):
        coords_in = [coords_in]
        filename = [filename]
        all_atom_feats = {k: [v] for k, v in all_atom_feats.items()}

    n_atoms_in = coords_in[0].shape[-2]
    is_bb_or_ca_pdb = n_atoms_in <= 4
    for i, c in enumerate(coords_in):
        n_res = c.shape[0]
        if isinstance(filename, list):
            fname = filename[i]
        elif write_to_frames or len(coords_in) == 1:
            fname = filename
        else:
            fname = f"{filename[:-4]}_{i}.pdb"

        if is_bb_or_ca_pdb:
            c_flat = rearrange(c, "n a c -> (n a) c")
            if n_atoms_in == 1:
                atoms = ["CA"]
            if n_atoms_in == 3:
                atoms = ["N", "CA", "C"]
            if n_atoms_in == 4:
                atoms = ["N", "CA", "C", "O"]
            pdb_str = bb_coords_to_pdb_str(c_flat, atoms)
        else:
            feats_i = {k: v[i][:n_res] for k, v in all_atom_feats.items()}
            pdb_str = feats_to_pdb_str(c, conect=conect, **feats_i)
        _write_pdb_string(pdb_str, fname, append=write_to_frames and i > 0)


###################### LOSSES ###################################


def masked_cross_entropy(logprobs, target, loss_mask):
    # target is onehot
    cel = -(target * logprobs)
    cel = cel * loss_mask[..., None]
    cel = cel.sum((-1, -2)) / loss_mask.sum(-1).clamp(min=1e-6)
    return cel


def masked_mse(x, y, mask, weight=None):
    data_dims = tuple(range(1, len(x.shape)))
    mse = (x - y).pow(2) * mask
    if weight is not None:
        mse = mse * expand(weight, mse)
    mse = mse.sum(data_dims) / mask.sum(data_dims).clamp(min=1e-6)
    return mse


###################### ALIGN ###################################


def quick_tmalign(
    p, p_sele, q_sele, tmscore_type="avg", differentiable_rmsd=False, rmsd_type="ca"
):
    # sota 210712
    write_coords_to_pdb(p_sele[:, 1:2], "temp_p.pdb", atoms=["CA"], batched=False)
    write_coords_to_pdb(q_sele[:, 1:2], "temp_q.pdb", atoms=["CA"], batched=False)
    cmd = f"{PATH_TO_TMALIGN} temp_p.pdb temp_q.pdb -m temp_matrix.txt"
    outputs = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

    # Get RMSD and TM scores
    tmout = outputs.stdout.split("\n")
    rmsd = float(tmout[16].split()[4][:-1])
    tmscore1 = float(tmout[17].split()[1])
    tmscore2 = float(tmout[18].split()[1])
    if tmscore_type == "avg":
        tmscore = (tmscore1 + tmscore2) / 2
    elif tmscore_type == "1" or tmscore_type == "query":
        tmscore = tmscore1
    elif tmscore_type == "2":
        tmscore = tmscore2
    elif tmscore_type == "both":
        tmscore = (tmscore1, tmscore2)

    # Get R, t and transform p coords
    m = open("temp_matrix.txt", "r").readlines()[2:5]
    m = [l.strip()[1:].strip() for l in m]
    m = torch.Tensor([[float(i) for i in l.split()] for l in m]).to(p_sele.device)
    R = m[:, 1:].t()
    t = m[:, 0]
    aligned_psele = p_sele @ R + t
    aligned = p @ R + t

    # Option 2 for rms - MSE of aligned against target coords using TMalign seq alignment. Differentiable
    if differentiable_rmsd:
        pi, qi = 0, 0
        p_idxs, q_idxs = [], []
        for i, c in enumerate(tmout[23]):
            if c in [":", "."]:
                p_idxs.append(pi)
                q_idxs.append(qi)
            if tmout[22][i] != "-":
                pi += 1
            if tmout[24][i] != "-":
                qi += 1
        tmalign_seq_p = p_sele[p_idxs]
        tmalign_seq_q = q_sele[q_idxs]
        if rmsd_type == "ca":
            tmalign_seq_p = tmalign_seq_p[:, 1]
            tmalign_seq_q = tmalign_seq_q[:, 1]
        elif rmsd_type == "bb":
            pass
        rmsd = (tmalign_seq_p - tmalign_seq_q).pow(2).sum(-1).sqrt().mean()

    # Delete temp files: p.pdb, q.pdb, matrix.txt, tmalign.out
    subprocess.run(shlex.split("rm temp_p.pdb"))
    subprocess.run(shlex.split("rm temp_q.pdb"))
    subprocess.run(shlex.split("rm temp_matrix.txt"))

    return {"aligned": aligned, "rmsd": rmsd, "tm_score": tmscore, "R": R, "t": t}


###################### OTHER ###################################


def expand(x, tgt=None, dim=1):
    if tgt is None:
        for _ in range(dim):
            x = x[..., None]
    else:
        while len(x.shape) < len(tgt.shape):
            x = x[..., None]
    return x


def hookfn(name, verbose=False):
    def f(grad):
        if check_nan_inf(grad) > 0:
            print(name, "grad nan/infs", grad.shape, check_nan_inf(grad), grad)
        if verbose:
            print(name, "grad shape", grad.shape, "norm", grad.norm())

    return f


def trigger_nan_check(name, x):
    if check_nan_inf(x) > 0:
        print(name, check_nan_inf(x))
        raise Exception


def check_nan_inf(x):
    return torch.isinf(x).sum() + torch.isnan(x).sum()


def directory_find(atom, root="."):
    for path, dirs, files in os.walk(root):
        if atom in dirs:
            return os.path.join(path, atom)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_config(path, return_dict=False):
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)
    if return_dict:
        return config, config_dict
    else:
        return config
