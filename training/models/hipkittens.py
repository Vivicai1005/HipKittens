import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.autograd import Function

import tk_kernel_fwd
import tk_kernel_bkwd


class HipKittensFlashAttnFn(Function):
    """
    Inputs/outputs are BNHD (batch, seq, heads, dim), like your harness.
    Forward:  O, L  via tk_kernel_fwd.dispatch_fwd
    Backward: dQ,dK,dV via tk_kernel_bkwd.{dispatch_prep,dispatch_bwd_combined,dispatch_dq_shuffle}
    Compute in bf16, save L and O for backward, return O in input dtype.
    """

    @staticmethod
    def forward(ctx, q_bnhd: torch.Tensor, k_bnhd: torch.Tensor, v_bnhd: torch.Tensor):
        B, N, H, D = q_bnhd.shape
        HKV = k_bnhd.shape[2]
        dev = q_bnhd.device
        out_dtype = q_bnhd.dtype  # remember caller dtype

        # Cast to bf16 for kernels (compute) and make contiguous
        q = q_bnhd.to(torch.bfloat16).contiguous()
        k = k_bnhd.to(torch.bfloat16).contiguous()
        v = v_bnhd.to(torch.bfloat16).contiguous()

        # Allocate outputs for the kernels
        O = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()    # BNHD
        L = torch.empty((B, H, 1, N), dtype=torch.float32,  device=dev).contiguous()    # BHN1 (matches your harness)

        # Forward kernel
        tk_kernel_fwd.dispatch_fwd(q, k, v, O, L)

        if O.isnan().any():
            print("O is nan")
            breakpoint()
        if L.isnan().any():
            print("L is nan")
            breakpoint()

        # Save for backward
        ctx.save_for_backward(q, k, v, O, L)
        # Return in caller dtype
        return O.to(out_dtype)

    @staticmethod
    def backward(ctx, dO_bnhd: torch.Tensor):
        q, k, v, O, L = ctx.saved_tensors
        B, N, H, D = O.shape
        HKV = k.shape[2]
        dev = dO_bnhd.device

        # Cast grad to bf16 for kernels
        dO = dO_bnhd.to(torch.bfloat16).contiguous()

        # Allocate grads and workspaces
        dQ_in = torch.empty((B, H, N, D), dtype=torch.bfloat16, device=dev).contiguous()  # BHND (pre-shuffle)
        dQ    = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHD
        dK    = torch.empty((B, N, HKV, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHD
        dV    = torch.empty((B, N, HKV, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHD
        delta = torch.empty((B, H, N, 1), dtype=torch.float32,  device=dev).contiguous()  # BH1N (matches your harness)

        if dO.isnan().any():
            print("dO is nan")
            breakpoint()

        # Backward kernels
        tk_kernel_bkwd.dispatch_prep(O, dO, delta)
        if delta.isnan().any():
            print("delta is nan")
            breakpoint()

        tk_kernel_bkwd.dispatch_bwd_combined(q, k, v, O, dO, dQ_in, dK, dV, L, delta)
        if dQ_in.isnan().any():
            print("dQ_in is nan")
            breakpoint()
        if dQ.isnan().any():
            print("dQ is nan")
            breakpoint()
        if dK.isnan().any():
            print("dK is nan")
            breakpoint()
        if dV.isnan().any():
            print("dV is nan")
            breakpoint()

        tk_kernel_bkwd.dispatch_dq_shuffle(dQ_in, dQ)
        if dQ.isnan().any():
            print("dQ is nan")
            breakpoint()

        return dQ.to(dO_bnhd.dtype), dK.to(dO_bnhd.dtype), dV.to(dO_bnhd.dtype)


class HipKittensBertSelfAttention(nn.Module):
    """
    Uses HipKittensFlashAttnFn when there is NO padding.
    Falls back to MHA-style expansion if num_key_value_heads < num_attention_heads (GQA).
    Expects HF additive mask: [B,1,1,N] (0 keep, -inf mask)
    """
    def __init__(self, config, layer_idx=None, deterministic: bool = False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError("hidden_size must be multiple of num_attention_heads")
        
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads                                        # h_q
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)  # h_kv
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        
        # Linear layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.num_attention_heads * self.attention_head_size)
        self.key   = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.deterministic = deterministic
        self.is_causal = False

        print(f"HipKittens BertSelfAttention layer {layer_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H = self.num_attention_heads
        HKV = self.num_key_value_heads
        D = self.attention_head_size

        q = self.query(hidden_states).view(B, N, H, D).to(torch.bfloat16).contiguous()
        k = self.key(hidden_states).view(B, N, HKV, D).to(torch.bfloat16).contiguous()
        v = self.value(hidden_states).view(B, N, HKV, D).to(torch.bfloat16).contiguous()

        out_bnhd = HipKittensFlashAttnFn.apply(q, k, v)  # BNHD
        ctx = out_bnhd.to(q.dtype).contiguous().view(B, N, H * D)
        return ctx, None

