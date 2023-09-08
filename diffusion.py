"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Noise and diffusion utils.
"""
from scipy.stats import norm
import torch
from torchtyping import TensorType

from core import utils


def noise_schedule(
    time: TensorType[float],
    function: str = "uniform",
    sigma_data: float = 10.0,
    psigma_mean: float = -1.2,
    psigma_std: float = 1.2,
    s_min: float = 0.001,
    s_max: float = 60,
    rho: float = 7.0,
    time_power: float = 4.0,
    constant_val: float = 0.0,
):
    def sampling_noise(time):
        # high noise = 1; low noise = 0. opposite of Karras et al. schedule
        term1 = s_max ** (1 / rho)
        term2 = (1 - time) * (s_min ** (1 / rho) - s_max ** (1 / rho))
        noise_level = sigma_data * ((term1 + term2) ** rho)
        return noise_level

    if function == "lognormal":
        normal_sample = torch.Tensor(norm.ppf(time.cpu())).to(time)
        noise_level = sigma_data * torch.exp(psigma_mean + psigma_std * normal_sample)
    elif function == "uniform":
        noise_level = sampling_noise(time)
    elif function == "mpnn":
        time = time**time_power
        noise_level = sampling_noise(time)
    elif function == "constant":
        noise_level = torch.ones_like(time) * constant_val
    return noise_level


def noise_coords(
    coords: TensorType["b n a x", float],
    noise_level: TensorType["b", float],
    dummy_fill_masked_atoms: bool = False,
    atom_mask: TensorType["b n a"] = None,
):
    # Does not apply atom mask after adding noise
    if dummy_fill_masked_atoms:
        assert atom_mask is not None
        dummy_fill_mask = 1 - atom_mask
        dummy_fill_value = coords[..., 1:2, :]  # CA
        # dummy_fill_value = utils.fill_in_cbeta_for_atom37(coords)[..., 3:4, :]  # CB
        coords = (
            coords * atom_mask[..., None]
            + dummy_fill_value * dummy_fill_mask[..., None]
        )

    noise = torch.randn_like(coords) * utils.expand(noise_level, coords)
    noisy_coords = coords + noise
    return noisy_coords
