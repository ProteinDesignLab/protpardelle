"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Neural network modules. Many of these are adapted from open source modules.
"""
from typing import List, Sequence, Optional

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import numpy as np
from rotary_embedding_torch import RotaryEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel

from core import protein_mpnn
from core import residue_constants
from core import utils


########################################
# Adapted from https://github.com/ermongroup/ddim


def downsample(x):
    return nn.functional.avg_pool2d(x, 2, 2, ceil_mode=True)


def upsample_coords(x, shape):
    new_l, new_w = shape
    return nn.functional.interpolate(x, size=(new_l, new_w), mode="nearest")


########################################
# Adapted from https://github.com/aqlaboratory/openfold


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.contiguous().permute(first_inds + [zero_index + i for i in inds])


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


class RelativePositionalEncoding(nn.Module):
    def __init__(self, attn_dim=8, max_rel_idx=32):
        super().__init__()
        self.max_rel_idx = max_rel_idx
        self.n_rel_pos = 2 * self.max_rel_idx + 1
        self.linear = nn.Linear(self.n_rel_pos, attn_dim)

    def forward(self, residue_index):
        d_ij = residue_index[..., None] - residue_index[..., None, :]
        v_bins = torch.arange(self.n_rel_pos).to(d_ij.device) - self.max_rel_idx
        idxs = (d_ij[..., None] - v_bins[None, None]).abs().argmin(-1)
        p_ij = nn.functional.one_hot(idxs, num_classes=self.n_rel_pos)
        embeddings = self.linear(p_ij.float())
        return embeddings


########################################
# Adapted from https://github.com/NVlabs/edm


class Noise_Embedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


########################################
# Adapted from github.com/lucidrains
# https://github.com/lucidrains/denoising-diffusion-pytorch
# https://github.com/lucidrains/recurrent-interface-network-pytorch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def posemb_sincos_1d(patches, temperature=10000, residue_index=None):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device=device) if residue_index is None else residue_index
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[..., None] * omega
    pe = torch.cat((n.sin(), n.cos()), dim=-1)
    return pe.type(dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class NoiseConditioningBlock(nn.Module):
    def __init__(self, n_in_channel, n_out_channel):
        super().__init__()
        self.block = nn.Sequential(
            Noise_Embedding(n_in_channel),
            nn.Linear(n_in_channel, n_out_channel),
            nn.SiLU(),
            nn.Linear(n_out_channel, n_out_channel),
            Rearrange("b d -> b 1 d"),
        )

    def forward(self, noise_level):
        return self.block(noise_level)


class TimeCondResnetBlock(nn.Module):
    def __init__(
        self, nic, noc, cond_nc, conv_layer=nn.Conv2d, dropout=0.1, n_norm_in_groups=4
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=nic // n_norm_in_groups, num_channels=nic),
            nn.SiLU(),
            conv_layer(nic, noc, 3, 1, 1),
        )
        self.cond_proj = nn.Linear(cond_nc, noc * 2)
        self.mid_norm = nn.GroupNorm(num_groups=noc // 4, num_channels=noc)
        self.dropout = dropout if dropout is None else nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=noc // 4, num_channels=noc),
            nn.SiLU(),
            conv_layer(noc, noc, 3, 1, 1),
        )
        self.mismatch = False
        if nic != noc:
            self.mismatch = True
            self.conv_match = conv_layer(nic, noc, 1, 1, 0)

    def forward(self, x, time=None):
        h = self.block1(x)

        if time is not None:
            h = self.mid_norm(h)
            scale, shift = self.cond_proj(time).chunk(2, dim=-1)
            h = (h * (utils.expand(scale, h) + 1)) + utils.expand(shift, h)

        if self.dropout is not None:
            h = self.dropout(h)

        h = self.block2(h)

        if self.mismatch:
            x = self.conv_match(x)

        return x + h


class TimeCondAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        heads=4,
        dim_head=32,
        norm=False,
        norm_context=False,
        time_cond_dim=None,
        attn_bias_dim=None,
        rotary_embedding_module=None,
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim * 2))

            nn.init.zeros_(self.time_cond[-1].weight)
            nn.init.zeros_(self.time_cond[-1].bias)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_bias_proj = None
        if attn_bias_dim is not None:
            self.attn_bias_proj = nn.Sequential(
                Rearrange("b a i j -> b i j a"),
                nn.Linear(attn_bias_dim, heads),
                Rearrange("b i j a -> b a i j"),
            )

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)

        self.use_rope = False
        if rotary_embedding_module is not None:
            self.use_rope = True
            self.rope = rotary_embedding_module

    def forward(self, x, context=None, time=None, attn_bias=None, seq_mask=None):
        # attn_bias is b, c, i, j
        h = self.heads
        has_context = exists(context)

        context = default(context, x)

        if x.shape[-1] != self.norm.gamma.shape[-1]:
            print(context.shape, x.shape, self.norm.gamma.shape)

        x = self.norm(x)

        if exists(time):
            scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if has_context:
            context = self.norm_context(context)

        if seq_mask is not None:
            x = x * seq_mask[..., None]

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        q = q * self.scale

        if self.use_rope:
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        if attn_bias is not None:
            if self.attn_bias_proj is not None:
                attn_bias = self.attn_bias_proj(attn_bias)
            sim += attn_bias
        if seq_mask is not None:
            attn_mask = torch.einsum("b i, b j -> b i j", seq_mask, seq_mask)[:, None]
            sim -= (1 - attn_mask) * 1e6
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if seq_mask is not None:
            out = out * seq_mask[..., None]
        return out


class TimeCondFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dim_out=None, time_cond_dim=None, dropout=0.1):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.norm = LayerNorm(dim)

        self.time_cond = None
        self.dropout = None
        inner_dim = int(dim * mult)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, inner_dim * 2),
            )

            nn.init.zeros_(self.time_cond[-1].weight)
            nn.init.zeros_(self.time_cond[-1].bias)

        self.linear_in = nn.Linear(dim, inner_dim)
        self.nonlinearity = nn.SiLU()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(inner_dim, dim_out)
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, x, time=None):
        x = self.norm(x)
        x = self.linear_in(x)
        x = self.nonlinearity(x)

        if exists(time):
            scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if exists(self.dropout):
            x = self.dropout(x)

        return self.linear_out(x)


class TimeCondTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        time_cond_dim,
        attn_bias_dim=None,
        mlp_inner_dim_mult=4,
        position_embedding_type: str = "rotary",
    ):
        super().__init__()

        self.rope = None
        self.pos_emb_type = position_embedding_type
        if position_embedding_type == "rotary":
            self.rope = RotaryEmbedding(dim=32)
        elif position_embedding_type == "relative":
            self.relpos = nn.Sequential(
                RelativePositionalEncoding(attn_dim=heads),
                Rearrange("b i j d -> b d i j"),
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TimeCondAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            norm=True,
                            time_cond_dim=time_cond_dim,
                            attn_bias_dim=attn_bias_dim,
                            rotary_embedding_module=self.rope,
                        ),
                        TimeCondFeedForward(
                            dim, mlp_inner_dim_mult, time_cond_dim=time_cond_dim
                        ),
                    ]
                )
            )

    def forward(
        self,
        x,
        time=None,
        attn_bias=None,
        context=None,
        seq_mask=None,
        residue_index=None,
    ):
        if self.pos_emb_type == "absolute":
            pos_emb = posemb_sincos_1d(x)
            x = x + pos_emb
        elif self.pos_emb_type == "absolute_residx":
            assert residue_index is not None
            pos_emb = posemb_sincos_1d(x, residue_index=residue_index)
            x = x + pos_emb
        elif self.pos_emb_type == "relative":
            assert residue_index is not None
            pos_emb = self.relpos(residue_index)
            attn_bias = pos_emb if attn_bias is None else attn_bias + pos_emb
        if seq_mask is not None:
            x = x * seq_mask[..., None]

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(
                x, context=context, time=time, attn_bias=attn_bias, seq_mask=seq_mask
            )
            x = x + ff(x, time=time)
            if seq_mask is not None:
                x = x * seq_mask[..., None]

        return x


class TimeCondUViT(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        dim: int,
        patch_size: int = 1,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        n_filt_per_layer: List[int] = [],
        n_blocks_per_layer: int = 2,
        n_atoms: int = 37,
        channels_per_atom: int = 6,
        attn_bias_dim: int = None,
        time_cond_dim: int = None,
        conv_skip_connection: bool = False,
        position_embedding_type: str = "rotary",
    ):
        super().__init__()

        # Initialize configuration params
        if time_cond_dim is None:
            time_cond_dim = dim * 4
        self.position_embedding_type = position_embedding_type
        channels = channels_per_atom
        self.n_conv_layers = n_conv_layers = len(n_filt_per_layer)
        if n_conv_layers > 0:
            post_conv_filt = n_filt_per_layer[-1]
        self.conv_skip_connection = conv_skip_connection and n_conv_layers == 1
        transformer_seq_len = seq_len // (2**n_conv_layers)
        assert transformer_seq_len % patch_size == 0
        num_patches = transformer_seq_len // patch_size
        dim_a = post_conv_atom_dim = max(1, n_atoms // (2 ** (n_conv_layers - 1)))
        if n_conv_layers == 0:
            patch_dim = patch_size * n_atoms * channels_per_atom
            patch_dim_out = patch_size * n_atoms * 3
            dim_a = n_atoms
        elif conv_skip_connection and n_conv_layers == 1:
            patch_dim = patch_size * (channels + post_conv_filt) * post_conv_atom_dim
            patch_dim_out = patch_size * post_conv_filt * post_conv_atom_dim
        elif n_conv_layers > 0:
            patch_dim = patch_dim_out = patch_size * post_conv_filt * post_conv_atom_dim

        # Make downsampling conv
        # Downsamples n-1 times where n is n_conv_layers
        down_conv = []
        block_in = channels
        for i, nf in enumerate(n_filt_per_layer):
            block_out = nf
            layer = []
            for j in range(n_blocks_per_layer):
                n_groups = 2 if i == 0 and j == 0 else 4
                layer.append(
                    TimeCondResnetBlock(
                        block_in, block_out, time_cond_dim, n_norm_in_groups=n_groups
                    )
                )
                block_in = block_out
            down_conv.append(nn.ModuleList(layer))
        self.down_conv = nn.ModuleList(down_conv)

        # Make transformer
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) a -> b n (p c a)", p=patch_size),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )
        self.transformer = TimeCondTransformer(
            dim,
            depth,
            heads,
            dim_head,
            time_cond_dim,
            attn_bias_dim=attn_bias_dim,
            position_embedding_type=position_embedding_type,
        )
        self.from_patch = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, patch_dim_out),
            Rearrange("b n (p c a) -> b c (n p) a", p=patch_size, a=dim_a),
        )
        nn.init.zeros_(self.from_patch[-2].weight)
        nn.init.zeros_(self.from_patch[-2].bias)

        # Make upsampling conv
        up_conv = []
        for i, nf in enumerate(reversed(n_filt_per_layer)):
            skip_in = nf
            block_out = nf
            layer = []
            for j in range(n_blocks_per_layer):
                layer.append(
                    TimeCondResnetBlock(block_in + skip_in, block_out, time_cond_dim)
                )
                block_in = block_out
            up_conv.append(nn.ModuleList(layer))
        self.up_conv = nn.ModuleList(up_conv)

        # Conv out
        if n_conv_layers > 0:
            self.conv_out = nn.Sequential(
                nn.GroupNorm(num_groups=block_out // 4, num_channels=block_out),
                nn.SiLU(),
                nn.Conv2d(block_out, channels // 2, 3, 1, 1),
            )

    def forward(
        self, coords, time_cond, pair_bias=None, seq_mask=None, residue_index=None
    ):
        if self.n_conv_layers > 0:  # pad up to even dims
            coords = F.pad(coords, (0, 0, 0, 0, 0, 1, 0, 0))

        x = rearr_coords = rearrange(coords, "b n a c -> b c n a")
        hiddens = []
        for i, layer in enumerate(self.down_conv):
            for block in layer:
                x = block(x, time=time_cond)
                hiddens.append(x)
            if i != self.n_conv_layers - 1:
                x = downsample(x)

        if self.conv_skip_connection:
            x = torch.cat([x, rearr_coords], 1)

        x = self.to_patch_embedding(x)
        # if self.position_embedding_type == 'absolute':
        #     pos_emb = posemb_sincos_1d(x)
        #     x = x + pos_emb
        if seq_mask is not None and x.shape[1] == seq_mask.shape[1]:
            x *= seq_mask[..., None]
        x = self.transformer(
            x,
            time=time_cond,
            attn_bias=pair_bias,
            seq_mask=seq_mask,
            residue_index=residue_index,
        )
        x = self.from_patch(x)

        for i, layer in enumerate(self.up_conv):
            for block in layer:
                x = torch.cat([x, hiddens.pop()], 1)
                x = block(x, time=time_cond)
            if i != self.n_conv_layers - 1:
                x = upsample_coords(x, hiddens[-1].shape[2:])

        if self.n_conv_layers > 0:
            x = self.conv_out(x)
            x = x[..., :-1, :]  # drop even-dims padding

        x = rearrange(x, "b c n a -> b n a c")
        return x


########################################


class LinearWarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_lr,
        warmup_steps=1000,
        decay_steps=int(1e6),
        min_lr=1e-6,
        **kwargs,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + decay_steps
        super(LinearWarmupCosineDecay, self).__init__(optimizer, **kwargs)

    def get_lr(self):
        # TODO double check for off-by-one errors
        if self.last_epoch < self.warmup_steps:
            curr_lr = self.last_epoch / self.warmup_steps * self.max_lr
            return [curr_lr for group in self.optimizer.param_groups]
        elif self.last_epoch < self.total_steps:
            time = (self.last_epoch - self.warmup_steps) / self.decay_steps * np.pi
            curr_lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + np.cos(time)
            )
            return [curr_lr for group in self.optimizer.param_groups]
        else:
            return [self.min_lr for group in self.optimizer.param_groups]


class NoiseConditionalProteinMPNN(nn.Module):
    def __init__(
        self,
        n_channel=128,
        n_layers=3,
        n_neighbors=32,
        time_cond_dim=None,
        vocab_size=21,
        input_S_is_embeddings=False,
    ):
        super().__init__()
        self.n_channel = n_channel
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.time_cond_dim = time_cond_dim
        self.vocab_size = vocab_size
        self.bb_idxs_if_atom37 = [
            residue_constants.atom_order[a] for a in ["N", "CA", "C", "O"]
        ]

        self.mpnn = protein_mpnn.ProteinMPNN(
            num_letters=vocab_size,
            node_features=n_channel,
            edge_features=n_channel,
            hidden_dim=n_channel,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            vocab=vocab_size,
            k_neighbors=n_neighbors,
            augment_eps=0.0,
            dropout=0.1,
            ca_only=False,
            time_cond_dim=time_cond_dim,
            input_S_is_embeddings=input_S_is_embeddings,
        )

    def forward(
        self, denoised_coords, noisy_aatype, seq_mask, residue_index, time_cond
    ):
        if denoised_coords.shape[-2] == 37:
            denoised_coords = denoised_coords[:, :, self.bb_idxs_if_atom37]

        node_embs, encoder_embs = self.mpnn(
            X=denoised_coords,
            S=noisy_aatype,
            mask=seq_mask,
            chain_M=seq_mask,
            residue_idx=residue_index,
            chain_encoding_all=seq_mask,
            randn=None,
            use_input_decoding_order=False,
            decoding_order=None,
            causal_mask=False,
            time_cond=time_cond,
            return_node_embs=True,
        )
        return node_embs, encoder_embs
