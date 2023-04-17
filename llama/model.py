# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.pjrt as pjrt
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,  # TODO: change these classes to the regular versions
    RowParallelLinear,
    ColumnParallelLinear,
)

import torch_xla.experimental.xla_sharding as xs
import numpy as np


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        #self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()  # TODO: this need to be modified for TPU
        self.n_local_heads = args.n_heads # // xm.xrt_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        # self.cache_k = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # )
        # self.cache_v = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # )
        # self.register_buffer("cache_k", torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # ))
        # self.register_buffer("cache_v", torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # ))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], input_idexes: torch.Tensor, cache_kv):
        bsz, seqlen, _ = x.shape
        cache_k, cache_v = cache_kv
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # xk, xv should be sharded along the hidden dim,
        # then with view, we reshape the last local_head dim to head_dim
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # TODO(yeounoh) column-wise sharding here
        #num_devices= pjrt.global_device_count()
        #col_mesh = xs.Mesh(np.arange(num_devices), (1, 1, num_devices, 1))
        #xs.mark_sharding(xq, col_mesh, (0,1,2,3))
        #xs.mark_sharding(xk, col_mesh, (0,1,2,3))
        #xs.mark_sharding(xv, col_mesh, (0,1,2,3))

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        # self.cache_k.index_copy_(1, input_idexes, xk)
        # self.cache_v.index_copy_(1, input_idexes, xv)
        # cache_k = cache_k.index_copy(1, input_idexes, xk).to(xm.xla_device())
        # cache_v = cache_v.index_copy(1, input_idexes, xv).to(xm.xla_device())

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]
        # keys = self.cache_k[:, :]
        # values = self.cache_v[:, :]
        keys = cache_k[:, :]
        values = cache_v[:, :]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output), (cache_k, cache_v)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], input_idexes: torch.Tensor, cache_kv):
        h, new_cache_kv = self.attention.forward(self.attention_norm(x), freqs_cis, mask, input_idexes, cache_kv)
        h = x + h
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, new_cache_kv


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        self.cache_kvs = []
        n_local_heads = params.n_heads # // xm.xrt_world_size()
        head_dim = params.dim // params.n_heads
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
            # total #heads in a shard -> n_local_heads, cache_k & v are stroing the
            # sharded versions.
            cache_k = torch.zeros(
                (params.max_batch_size, params.max_seq_len, n_local_heads, head_dim)
            )#.to(xm.xla_device())
            cache_v = torch.zeros(
                (params.max_batch_size, params.max_seq_len, n_local_heads, head_dim)
            )#.to(xm.xla_device())
            # TODO(yeounoh) column-wise sharding here
            #num_devices= pjrt.global_device_count()
            #col_mesh = xs.Mesh(np.arange(num_devices), (1, 1, num_devices, 1))
            #xs.mark_sharding(cache_k, col_mesh, (0,1,2,3))
            #xs.mark_sharding(cache_v, col_mesh, (0,1,2,3))
            self.cache_kvs.append((cache_k, cache_v))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.register_buffer("freqs_cis", freqs_cis)

        mask = torch.full((1, 1, self.params.max_seq_len, self.params.max_seq_len), float("-inf")).to(torch.float)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, input_idexes: torch.Tensor, output_idex: torch.Tensor, cache_kvs):
        bsz, seqlen = tokens.shape
        assert bsz == self.params.max_batch_size
        # print(tokens)
        h = self.tok_embeddings(tokens)
        # self.freqs_cis = self.freqs_cis.to(h.device)
        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        freqs_cis = self.freqs_cis.index_select(0, input_idexes)

        # mask = None
        # if seqlen > 1:
            # mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            # mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        mask = self.mask.index_select(2, input_idexes)

        new_cache_kvs = []
        for layer, cache_kv in zip(self.layers, cache_kvs):
            h, new_cache_kv = layer(h, freqs_cis, mask, input_idexes, cache_kv)
            new_cache_kvs.append(new_cache_kv)
        h = self.norm(h)
        h = h.index_select(1, output_idex - input_idexes[0]).squeeze(dim=1)
        # output = self.output(h[:, -1, :])  # only compute last logits
        output = self.output(h)
        return output.float(), new_cache_kvs

