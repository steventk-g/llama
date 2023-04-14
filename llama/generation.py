# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import time
from typing import List

import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

from llama.tokenizer import Tokenizer
from llama.model import Transformer

import numpy as np
import torch_xla.experimental.xla_sharding as xs
import torch_xla.experimental.pjrt as pjrt


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._generate_one_token_fn = self._generate_one_token
        #self._generate_one_token_fn = torch.compile(backend="torchxla_trace_once")(self._generate_one_token)
        self.model = torch.compile(self.model, backend="torchxla_trace_once")
        self.num_devices = pjrt.global_device_count()

    def _generate_one_token(self, tokens, input_tokens, input_text_mask, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs, temperature, top_p):
        logits, cache_kvs = self.model(input_tokens, input_pos_tensor, output_pos_tensor, cache_kvs)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        input_text_mask_tmp = input_text_mask.index_select(1, cur_pos_tensor).squeeze(dim=1)
        tokens_tmp = tokens.index_select(1, cur_pos_tensor).squeeze(dim=1)
        next_token = torch.where(
            input_text_mask_tmp, tokens_tmp, next_token
        )
        # tokens[:, cur_pos] = next_token
        next_token = next_token.unsqueeze(1)
        tokens.index_copy_(1, cur_pos_tensor, next_token)
        input_pos_tensor = input_pos_tensor[-1:] + 1
        cur_pos_tensor += 1
        output_pos_tensor = cur_pos_tensor - 1
        input_tokens = tokens.index_select(1, input_pos_tensor)

        return tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        start_time = time.time()
        input_prepare_start_time = time.time()

        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=False, eos=False) for x in prompts]  # the hacked tokenizer don't have bos

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        # total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        total_len = params.max_seq_len

        # tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()  # TODO: this line puts input to cuda device
        tokens = torch.full((params.max_batch_size, total_len), self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        device = xm.xla_device()
        tokens = tokens.to(device)
        input_text_mask = tokens != self.tokenizer.pad_id
        # start_pos = min_prompt_size
        start_pos = 1
        cur_pos_tensor = torch.tensor(start_pos).to(device)
        input_pos_tensor = torch.arange(0, start_pos).to(device)
        output_pos_tensor = cur_pos_tensor - 1
        input_tokens = tokens.index_select(1, input_pos_tensor)
        cache_kvs = self.model.cache_kvs
        xm.mark_step(wait=True)
        print(f"Input prepared in {time.time() - input_prepare_start_time:.5f} seconds")

        num_devices = self.num_devices
        col_mesh = xs.Mesh(np.arange(num_devices), (1, 1, num_devices, 1))

        decoding_start_time = time.time()
        sharding_times = []
        for _ in range(start_pos, total_len):
            token_start_time = time.time()
            # TODO(yeounoh) shard cache_kv before token generation
            for i in range(len(cache_kvs)):
              for t in cache_kvs[i]:
                xs.mark_sharding(t, col_mesh, (0,1,2,3))
            sharding_times.append(time.time() - token_start_time)
            print(f"Sharded cache KVs in {sharding_times[-1]:.5f} seconds")
            # with xp.Trace('trace_generate_one_token'):
            tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs = self._generate_one_token_fn(tokens, input_tokens, input_text_mask, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs, temperature, top_p)
            xm.mark_step()
            print(f"Generated 1 token in {time.time() - token_start_time:.5f} seconds")
        self.model.cache_kvs = cache_kvs
        print(f"Decoded in {time.time() - decoding_start_time:.5f} seconds")
        print(f"Spent {sum(sharding_times):.5f} seconds on sharding cache KVs")
        

        output_prepare_start_time = time.time()
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            if i >= len(prompt_tokens):
                break
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        print(f"Detokenized ouput in {time.time() - output_prepare_start_time:.2f} seconds")
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    # probs_sort[mask] = 0.0
    probs_sort = torch.where(mask, 0.0, probs_sort)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
