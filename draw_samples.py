"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Entry point for unconditional or simple conditional sampling.
"""
import argparse
from datetime import datetime
import json
import os
import shlex
import subprocess
import sys
import time

from einops import repeat
import torch

from core import data
from core import residue_constants
from core import utils
import diffusion
import models
import sampling


def draw_and_save_samples(
    model,
    samples_per_len=8,
    lengths=range(50, 512),
    save_dir="./",
    mode="backbone",
    **sampling_kwargs,
):
    device = model.device
    if mode == "backbone":
        total_sampling_time = 0
        for l in lengths:
            prot_lens = torch.ones(samples_per_len).long() * l
            seq_mask = model.make_seq_mask_for_sampling(prot_lens=prot_lens)
            aux = sampling.draw_backbone_samples(
                model,
                seq_mask=seq_mask,
                pdb_save_path=f"{save_dir}/len{format(l, '03d')}_samp",
                return_aux=True,
                return_sampling_runtime=True,
                **sampling_kwargs,
            )
            total_sampling_time += aux["runtime"]
            print("Samples drawn for length", l)
        return total_sampling_time
    elif mode == "allatom":
        total_sampling_time = 0
        for l in lengths:
            prot_lens = torch.ones(samples_per_len).long() * l
            seq_mask = model.make_seq_mask_for_sampling(prot_lens=prot_lens)
            aux = sampling.draw_allatom_samples(
                model,
                seq_mask=seq_mask,
                pdb_save_path=f"{save_dir}/len{format(l, '03d')}",
                return_aux=True,
                **sampling_kwargs,
            )
            total_sampling_time += aux["runtime"]
            print("Samples drawn for length", l)
        return total_sampling_time


def parse_idx_string(idx_str):
    spans = idx_str.split(",")
    idxs = []
    for s in spans:
        if "-" in s:
            start, stop = s.split("-")
            idxs.extend(list(range(int(start), int(stop))))
        else:
            idxs.append(int(s))
    return idxs


class Manager(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter
        )

        self.parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="checkpoints",
            help="Path to denoiser model weights and config",
        )
        self.parser.add_argument(
            "--mpnnpath",
            type=str,
            default="checkpoints/minimpnn_state_dict.pth",
            help="Path to minimpnn model weights",
        )
        self.parser.add_argument(
            "--modeldir",
            type=str,
            help="Model base directory, ex 'training_logs/other/lemon-shape-51'",
        )
        self.parser.add_argument("--modelepoch", type=int, help="Model epoch, ex 1000")
        self.parser.add_argument(
            "--type", type=str, default="allatom", help="Type of model"
        )
        self.parser.add_argument(
            "--param", type=str, default=None, help="Which sampling param to vary"
        )
        self.parser.add_argument(
            "--paramval", type=str, default=None, help="Which param val to use"
        )
        self.parser.add_argument(
            "--parampath",
            type=str,
            default=None,
            help="Path to json file with params, either use param/paramval or parampath, not both",
        )
        self.parser.add_argument(
            "--perlen", type=int, default=2, help="How many samples per sequence length"
        )
        self.parser.add_argument(
            "--minlen", type=int, default=50, help="Minimum sequence length"
        )
        self.parser.add_argument(
            "--maxlen",
            type=int,
            default=60,
            help="Maximum sequence length, not inclusive",
        )
        self.parser.add_argument(
            "--steplen",
            type=int,
            default=5,
            help="How frequently to select sequence length, for steplen 2, would be 50, 52, 54, etc",
        )
        self.parser.add_argument(
            "--num_lens",
            type=int,
            required=False,
            help="If steplen not provided, how many random lengths to sample at",
        )
        self.parser.add_argument(
            "--targetdir", type=str, default=".", help="Directory to save results"
        )
        self.parser.add_argument(
            "--input_pdb", type=str, required=False, help="PDB file to condition on"
        )
        self.parser.add_argument(
            "--resample_idxs",
            type=str,
            required=False,
            help="Indices from PDB file to resample. Zero-indexed, comma-delimited, can use dashes, eg 0,2-5,7",
        )

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        self.args = self.parser.parse_args()

        return self.args


def main():
    # Set up params, arguments, sampling config
    ####################
    manager = Manager()
    manager.parse_args()
    args = manager.args
    print(args)
    is_test_run = False
    seed = 0
    samples_per_len = args.perlen
    min_len = args.minlen
    max_len = args.maxlen
    len_step_size = args.steplen
    device = "cuda:0"

    # setting default sampling config
    if args.type == "backbone":
        sampling_config = sampling.default_backbone_sampling_config()
    elif args.type == "allatom":
        sampling_config = sampling.default_allatom_sampling_config()

    sampling_kwargs = vars(sampling_config)

    # Parse conditioning inputs
    input_pdb_len = None
    if args.input_pdb:
        input_feats = utils.load_feats_from_pdb(args.input_pdb, protein_only=True)
        input_pdb_len = input_feats["aatype"].shape[0]
        if args.resample_idxs:
            print(
                f"Warning: when sampling conditionally, the input pdb length ({input_pdb_len} residues) is used automatically for the sampling lengths."
            )
            resample_idxs = parse_idx_string(args.resample_idxs)
        else:
            resample_idxs = list(range(input_pdb_len))
        cond_idxs = [i for i in range(input_pdb_len) if i not in resample_idxs]
        to_batch_size = lambda x: repeat(x, "... -> b ...", b=samples_per_len).to(
            device
        )

        # For unconditional model, center coords on whole structure
        centered_coords = data.apply_random_se3(
            input_feats["atom_positions"],
            atom_mask=input_feats["atom_mask"],
            translation_scale=0.0,
        )
        cond_kwargs = {}
        cond_kwargs["gt_coords"] = to_batch_size(centered_coords)
        cond_kwargs["gt_cond_atom_mask"] = to_batch_size(input_feats["atom_mask"])
        cond_kwargs["gt_cond_atom_mask"][:, resample_idxs] = 0
        cond_kwargs["gt_aatype"] = to_batch_size(input_feats["aatype"])
        cond_kwargs["gt_cond_seq_mask"] = torch.zeros_like(cond_kwargs["gt_aatype"])
        cond_kwargs["gt_cond_seq_mask"][:, cond_idxs] = 1
        sampling_kwargs.update(cond_kwargs)

    # Determine lengths to sample at
    if min_len is not None and max_len is not None:
        if len_step_size is not None:
            sampling_lengths = range(min_len, max_len, len_step_size)
        else:
            sampling_lengths = list(
                torch.randint(min_len, max_len, size=(args.num_lens,))
            )
    elif input_pdb_len is not None:
        sampling_lengths = [input_pdb_len]
    else:
        raise Exception("Need to provide a set of protein lengths or an input pdb.")

    total_num_samples = len(list(sampling_lengths)) * samples_per_len

    model_directory = args.modeldir
    epoch = args.modelepoch
    base_dir = args.targetdir

    date_string = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    if is_test_run:
        date_string = f"test-{date_string}"

    # Update sampling config with arguments
    if args.param:
        var_param = args.param
        var_value = args.paramval
        sampling_kwargs[var_param] = (
            None
            if var_value == "None"
            else int(var_value)
            if var_param == "n_steps"
            else float(var_value)
        )
    elif args.parampath:
        with open(args.parampath) as f:
            var_params = json.loads(f.read())
            sampling_kwargs.update(var_params)

    # this is only used for the readme, keep s_min and s_max as params instead of struct_noise_schedule
    sampling_kwargs_readme = list(sampling_kwargs.items())

    print("Base directory:", base_dir)
    save_dir = f"{base_dir}/samples"
    save_init_dir = f"{base_dir}/samples_inits"

    print("Samples saved to:", save_dir)
    ####################

    torch.manual_seed(seed)
    if not os.path.exists(save_dir):
        subprocess.run(shlex.split(f"mkdir -p {save_dir}"))

    if not os.path.exists(save_init_dir):
        subprocess.run(shlex.split(f"mkdir -p {save_init_dir}"))

    # Load model
    if args.type == "backbone":
        if args.model_checkpoint:
            checkpoint = f"{args.model_checkpoint}/backbone_state_dict.pth"
            cfg_path = f"{args.model_checkpoint}/backbone_pretrained.yml"
        else:
            checkpoint = (
                f"{model_directory}/checkpoints/epoch{epoch}_training_state.pth"
            )
            cfg_path = f"{model_directory}/configs/backbone.yml"
        config = utils.load_config(cfg_path)
        weights = torch.load(checkpoint, map_location=device)["model_state_dict"]
        model = models.Protpardelle(config, device=device)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()
        model.device = device
    elif args.type == "allatom":
        if args.model_checkpoint:
            checkpoint = f"{args.model_checkpoint}/allatom_state_dict.pth"
            cfg_path = f"{args.model_checkpoint}/allatom_pretrained.yml"
        else:
            checkpoint = (
                f"{model_directory}/checkpoints/epoch{epoch}_training_state.pth"
            )
            cfg_path = f"{model_directory}/configs/allatom.yml"
        config = utils.load_config(cfg_path)
        weights = torch.load(checkpoint, map_location=device)["model_state_dict"]
        model = models.Protpardelle(config, device=device)
        model.load_state_dict(weights)
        model.load_minimpnn(args.mpnnpath)
        model.to(device)
        model.eval()
        model.device = device

    if config.train.home_dir == '':
        config.train.home_dir = os.path.dirname(os.getcwd())

    # Sampling
    with open(save_dir + "/readme.txt", "w") as f:
        f.write(f"Sampling run for {date_string}\n")
        f.write(f"Random seed {seed}\n")
        f.write(f"Model checkpoint: {checkpoint}\n")
        f.write(
            f"{samples_per_len} samples per length from {min_len}:{max_len}:{len_step_size}\n"
        )
        f.write("Sampling params:\n")
        for k, v in sampling_kwargs_readme:
            f.write(f"{k}\t{v}\n")

    print(f"Model loaded from {checkpoint}")
    print(f"Beginning sampling for {date_string}...")

    # Draw samples
    start_time = time.time()
    sampling_time = draw_and_save_samples(
        model,
        samples_per_len=samples_per_len,
        lengths=sampling_lengths,
        save_dir=save_dir,
        mode=args.type,
        **sampling_kwargs,
    )
    time_elapsed = time.time() - start_time

    print(f"Sampling concluded after {time_elapsed} seconds.")
    print(f"Of this, {sampling_time} seconds were for actual sampling.")
    print(f"{total_num_samples} total samples were drawn.")

    with open(save_dir + "/readme.txt", "a") as f:
        f.write(f"Total job time: {time_elapsed} seconds\n")
        f.write(f"Model run time: {sampling_time} seconds\n")
        f.write(f"Total samples drawn: {total_num_samples}\n")

    return


if __name__ == "__main__":
    main()
