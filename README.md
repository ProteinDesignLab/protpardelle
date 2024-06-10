# protpardelle

Code for the paper: [An all-atom protein generative model](https://www.biorxiv.org/content/10.1101/2023.05.24.542194v1.full).

The code is under active development and we welcome contributions, feature requests, issues, corrections, and any questions! Where we have used or adapted code from others we have tried to give proper attribution, but please let us know if anything should be corrected.


![twitter_movie3](https://github.com/ProteinDesignLab/protpardelle/assets/16140426/98ed76c4-114b-4fa7-ae8a-e661082c8cdf)


## Environment and setup

To set up the conda environment, run `conda env create -f configs/environment.yml` then `conda activate delle`. You will also need to clone the [ProteinMPNN repository](https://github.com/dauparas/ProteinMPNN) to the same directory that contains the `protpardelle/` repository. You may also need to set the `home_dir` variable in the configs you use to the path to the directory containing the `protpardelle/` directory.


## Use in WebApp and Pymol

You can use protpardelle directly in a convenient HuggingFace Webapp powered by Gradio. 

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/ProteinDesignLab/protpardelle)

![HuggingFace Webapp](https://i.imgur.com/JZTMPb1.png)

Alternatively you can directly design proteins within Pymol. 
For this download the `protpardelle_pymol.py` file to your computer. 

Then open Pymol, and navigate to the directory where the file is located. 

```
pwd (to know which directory you are in)
cd path/to/dir
```
then load the file. If launching for the first time this will install `gradio_client` in the python installation that Pymol uses. This might take a few seconds and the Pymol window will appear frozen.

```
load protpardelle_pymol.py
```

**Conditional Design**

Currently conditional design codes are not working, we will fix the bugs soon; sorry for inconvenience.

To run conditional design first load a structure e.g
```
fetch 1pga
```

then select some residues (and optionally name the selection).
To generate 5 samples resampling the residues in a selection named `sele` run the following command in the Pymol console:
```
protpardelle 1pga, sele, 5
```

**Unconditional Design** 

To sample proteins between length 50 and 60 with step size 5 and generate 1 sample per sampled length use the following command:

```
protpardelle_uncond 50, 60, 5, 1
```

To use the backbone only model use:

```
protpardelle_uncond 50,60,5,1,backbone
```

## Inference

24/06/10 Update
1. We've updated a new backbone model weight and the config. We recommend using the new weight.
2. You can change some sampling hyperparameters in configs/sampling.yml. The most important hyperparameters are step_scale and s_churn. Increasing step scale (like 1.1 or 1.2) will generate more designable sample but less diverse samples, and decreasing s_churn will generate less designable sample but more diverse sample. You can also change n_steps, rho, s_max, and s_min, and we'd recommend reading karras paper(https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html) to understand the role of these hyperparameters.
3. Currently conditional generation is not working; we will be fixing it soon. 

The entry point for sampling is `draw_samples.py`, which is a convenience wrapper around `sampling.py` and the `model.sample()` function. There are a number of arguments which can be passed to control the model checkpoints, the sampling configuration, and lengths of the proteins sampled. Model weights are provided for both the backbone-only and all-atom versions of Protpardelle. Both of these are trained unconditionally; we will release conditional models in a later update. Below are some examples of how to draw samples.

The default command used to draw all-atom samples (for example, for 2 proteins at each length in `range(80, 100, 5)`):

`python draw_samples.py --type allatom --minlen 80 --maxlen 100 --steplen 5 --perlen 2`

We have also added the ability to provide an input PDB file and a list of (zero-indexed) indices to condition on from the PDB file. Note also that current models are single-chain only, so multi-chain PDBs will be treated as single chains (we intend to release multi-chain models in a later update). We can expect it to do better or worse depending on the problem (better on easier problems such as inpainting, worse on difficult problems such as discontiguous scaffolding). Use this command to resample the first 25 and 71st to 80th residues of `my_pdb.pdb`.

`python draw_samples.py --input_pdb my_pdb.pdb --resample_idxs 0-25,70-80`

For more control over the sampling process, including tweaking the sampling hyperparameters and more specific methods of conditioning, you can directly interface with the `model.sample()` function; we have provided examples of how to configure and run these commands in `sampling.py`.

## Training

Note (Sep 2023): the lab has decided to collect usage statistics on people interested in training their own versions of Protpardelle (for funding and other purposes). To obtain a copy of the repository with training code, please complete [this Google Form](https://docs.google.com/forms/d/1WKMVbydLh6LIegc3HfwMQhgL2_qnrY7ks9FM_ylo4ts) - you will receive a link to a Google Drive zip which contains the repository with training code. After publication, the plan is to include the full training code directly in this repository.

Pretrained model weights are provided, but if you are interested in training your own models, we have provided training code together with some basic online evaluation. You will need to create a Weights & Biases account.

The dataset can be downloaded from [CATH](http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/), and the train/validation/test splits used can be downloaded with

`wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json`

Some PDBs in these splits have since become obsolete; we manually replaced these PDBs with the files for the updated/new PDB IDs. The dataloader expects text files in the dataset directory named 'train_pdb_keys.list', 'eval_pdbs_keys.list', and 'test_pdb_keys.list' which list the filenames associated with each dataset split. This, together with the directory of PDB files, is sufficient for the dataloader.

The main entry point is `train.py`; there are some arguments to control computation, experimenting, etc. Model-specific training code is kept separate from the training infrastructure and handled by the runner classes in `runners.py`; model-related hyperparameters are handled by the config file. Using `configs/backbone.yml` trains a backbone-only model; `configs/allatom.yml` trains an all-atom model, and `configs/seqdes.yml` trains a mini-MPNN model. Some examples:

The default command (used to produce the saved model weights):

`python train.py  --project protpardelle --train --config configs/allatom.yml --num_workers 8`

For a simple debugging run for the mini-MPNN model:

`python train.py --config configs/seqdes.yml`

To overfit to 100 data examples using 8 dataloading workers for a crop-conditional backbone model with 2 layers, in `configs/backbone.yml` change `train.crop_conditional` and `model.crop_conditional` to True, and then run:

`python train.py --train --config configs/backbone.yml --overfit 100 --num_workers 8`

Training with DDP is a bit more involved and uses torch.distributed. Note that the batch size in the config becomes the per-device batch size. To train all-atom with DDP on 2 GPUs on a single node, run:

`python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --master_port=$RANDOM train.py --config configs/allatom.yml --train --n_gpu_per_node 2 --use_ddp --num_workers 8`

## Citation

If you find our work helpful, please cite

```
@article {chu2023allatom,
    author = {Alexander E. Chu and Lucy Cheng and Gina El Nesr and Minkai Xu and Po-Ssu Huang},
    title = {An all-atom protein generative model},
    year = {2023},
    doi = {10.1101/2023.05.24.542194},
    URL = {https://www.biorxiv.org/content/early/2023/05/25/2023.05.24.542194},
    journal = {bioRxiv}
}
```

