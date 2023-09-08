"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Dataloader from PDB files.
"""
import copy
import pickle
import json
import numpy as np
import torch
from torch.utils import data

from core import utils
from core import protein
from core import residue_constants


FEATURES_1D = (
    "coords_in",
    "torsions_in",
    "b_factors",
    "atom_positions",
    "aatype",
    "atom_mask",
    "residue_index",
    "chain_index",
)
FEATURES_FLOAT = (
    "coords_in",
    "torsions_in",
    "b_factors",
    "atom_positions",
    "atom_mask",
    "seq_mask",
)
FEATURES_LONG = ("aatype", "residue_index", "chain_index", "orig_size")


def make_fixed_size_1d(data, fixed_size=128):
    data_len = data.shape[0]
    if data_len >= fixed_size:
        extra_len = data_len - fixed_size
        start_idx = np.random.choice(np.arange(extra_len + 1))
        new_data = data[start_idx : (start_idx + fixed_size)]
        mask = torch.ones(fixed_size)
    if data_len < fixed_size:
        pad_size = fixed_size - data_len
        extra_shape = data.shape[1:]
        new_data = torch.cat([data, torch.zeros(pad_size, *extra_shape)], 0)
        mask = torch.cat([torch.ones(data_len), torch.zeros(pad_size)], 0)
    return new_data, mask


def apply_random_se3(coords_in, atom_mask=None, translation_scale=1.0):
    # unbatched. center on the mean of CA coords
    coords_mean = coords_in[:, 1:2].mean(-3, keepdim=True)
    coords_in -= coords_mean
    random_rot, _ = torch.linalg.qr(torch.randn(3, 3))
    coords_in = coords_in @ random_rot
    random_trans = torch.randn_like(coords_mean) * translation_scale
    coords_in += random_trans
    if atom_mask is not None:
        coords_in = coords_in * atom_mask[..., None]
    return coords_in


def get_masked_coords_array(coords, atom_mask):
    ma_mask = repeat(1 - atom_mask[..., None].cpu().numpy(), "... 1 -> ... 3")
    return np.ma.array(coords.cpu().numpy(), mask=ma_mask)


def make_crop_cond_mask_and_recenter_coords(
    atom_mask,
    atom_coords,
    contiguous_prob=0.05,
    discontiguous_prob=0.9,
    sidechain_only_prob=0.8,
    max_span_len=10,
    max_discontiguous_res=8,
    dist_threshold=8.0,
    recenter_coords=True,
):
    b, n, a = atom_mask.shape
    device = atom_mask.device
    seq_mask = atom_mask[..., 1]
    n_res = seq_mask.sum(-1)
    masks = []

    for i, nr in enumerate(n_res):
        nr = nr.int().item()
        mask = torch.zeros((n, a), device=device)
        conditioning_type = torch.distributions.Categorical(
            torch.tensor(
                [
                    contiguous_prob,
                    discontiguous_prob,
                    1.0 - contiguous_prob - discontiguous_prob,
                ]
            )
        ).sample()
        conditioning_type = ["contiguous", "discontiguous", "none"][conditioning_type]

        if conditioning_type == "contiguous":
            span_len = torch.randint(
                1, min(max_span_len, nr), (1,), device=device
            ).item()
            span_start = torch.randint(0, nr - span_len, (1,), device=device)
            mask[span_start : span_start + span_len, :] = 1
        elif conditioning_type == "discontiguous":
            # Extract CB atoms coordinates for the i-th example
            cb_atoms = atom_coords[i, :, 3]
            # Pairwise distances between CB atoms
            cb_distances = torch.cdist(cb_atoms, cb_atoms)
            close_mask = (
                cb_distances <= dist_threshold
            )  # Mask for selecting close CB atoms

            random_residue = torch.randint(0, nr, (1,), device=device).squeeze()
            cb_dist_i = cb_distances[random_residue] + 1e3 * (1 - seq_mask[i])
            close_mask = cb_dist_i <= dist_threshold
            n_neighbors = close_mask.sum().int()

            # pick how many neighbors (up to 10)
            n_sele = torch.randint(
                2,
                n_neighbors.clamp(min=3, max=max_discontiguous_res + 1),
                (1,),
                device=device,
            )

            # Select the indices of CB atoms that are close together
            idxs = torch.arange(n, device=device)[close_mask.bool()]
            idxs = idxs[torch.randperm(len(idxs))[:n_sele]]

            if len(idxs) > 0:
                mask[idxs] = 1

            if np.random.uniform() < sidechain_only_prob:
                mask[:, :5] = 0

        masks.append(mask)

    crop_cond_mask = torch.stack(masks)
    crop_cond_mask = crop_cond_mask * atom_mask
    if recenter_coords:
        motif_masked_array = get_masked_coords_array(atom_coords, crop_cond_mask)
        cond_coords_center = motif_masked_array.mean((1, 2))
        motif_mask = torch.Tensor(1 - cond_coords_center.mask).to(crop_cond_mask)
        means = torch.Tensor(cond_coords_center.data).to(atom_coords) * motif_mask
        coords_out = atom_coords - rearrange(means, "b c -> b 1 1 c")
    else:
        coords_out = atom_coords
    return coords_out, crop_cond_mask


class Dataset(data.Dataset):
    """Loads and processes PDBs into tensors."""

    def __init__(
        self,
        pdb_path,
        fixed_size,
        mode="train",
        overfit=-1,
        short_epoch=False,
        se3_data_augment=True,
    ):
        self.pdb_path = pdb_path
        self.fixed_size = fixed_size
        self.mode = mode
        self.overfit = overfit
        self.short_epoch = short_epoch
        self.se3_data_augment = se3_data_augment

        with open(f"{self.pdb_path}/{mode}_pdb_keys.list") as f:
            self.pdb_keys = np.array(f.read().split("\n")[:-1])

        if overfit > 0:
            n_data = len(self.pdb_keys)
            self.pdb_keys = np.random.choice(
                self.pdb_keys, min(n_data, overfit), replace=False
            ).repeat(n_data // overfit)

    def __len__(self):
        if self.short_epoch:
            return min(len(self.pdb_keys), 256)
        else:
            return len(self.pdb_keys)

    def __getitem__(self, idx):
        pdb_key = self.pdb_keys[idx]
        data = self.get_item(pdb_key)
        # For now, replace dataloading errors with a random pdb. 10 tries
        for _ in range(10):
            if data is not None:
                return data
            pdb_key = self.pdb_keys[np.random.randint(len(self.pdb_keys))]
            data = self.get_item(pdb_key)
        raise Exception("Failed to load data example after 10 tries.")

    def get_item(self, pdb_key):
        example = {}

        if self.pdb_path.endswith("cath_s40_dataset"):  # CATH pdbs
            data_file = f"{self.pdb_path}/dompdb/{pdb_key}"
        elif self.pdb_path.endswith("ingraham_cath_dataset"):  # ingraham splits
            data_file = f"{self.pdb_path}/pdb_store/{pdb_key}"
        else:
            raise Exception("Invalid pdb path.")

        try:
            example = utils.load_feats_from_pdb(data_file)
            coords_in = example["atom_positions"]
        except FileNotFoundError:
            raise Exception(f"File {pdb_key} not found. Check if dataset is corrupted?")
        except RuntimeError:
            return None

        # Apply data augmentation
        if self.se3_data_augment:
            coords_in = apply_random_se3(coords_in, atom_mask=example["atom_mask"])

        orig_size = coords_in.shape[0]
        example["coords_in"] = coords_in
        example["orig_size"] = torch.ones(1) * orig_size

        fixed_size_example = {}
        seq_mask = None
        for k, v in example.items():
            if k in FEATURES_1D:
                fixed_size_example[k], seq_mask = make_fixed_size_1d(
                    v, fixed_size=self.fixed_size
                )
            else:
                fixed_size_example[k] = v
        if seq_mask is not None:
            fixed_size_example["seq_mask"] = seq_mask

        example_out = {}
        for k, v in fixed_size_example.items():
            if k in FEATURES_FLOAT:
                example_out[k] = v.float()
            elif k in FEATURES_LONG:
                example_out[k] = v.long()

        return example_out

    def collate(self, example_list):
        out = {}
        for ex in example_list:
            for k, v in ex.items():
                out.setdefault(k, []).append(v)
        return {k: torch.stack(v) for k, v in out.items()}

    def sample(self, n=1, return_data=True, return_keys=False):
        keys = self.pdb_keys[torch.randperm(self.__len__())[:n].long()]

        if return_keys and not return_data:
            return keys

        if n == 1:
            data = self.collate([self.get_item(keys)])
        else:
            data = self.collate([self.get_item(key) for key in keys])

        if return_data and return_keys:
            return data, keys
        if return_data and not return_keys:
            return data
