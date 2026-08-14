

from typing import List, Optional

import torch
from torch import nn

from CompleteBin.logger import get_logger

logger = get_logger()


def encode_seq2vec(seq_tensor):
    sft = torch.softmax(torch.mean(seq_tensor, dim=-1, keepdim=True), dim=1)
    seq_rep = torch.sum(seq_tensor * sft, dim=1)  # [B, C]
    return seq_rep


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: int,  p: float):
        super().__init__()
        
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(p)
        self.dropout_o = nn.Dropout(p)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # print(x.shape)
        x = self.dropout(x)
        x = self.down_proj(x)
        # print(x.shape)
        return self.dropout_o(x)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def compute_linear_scaling_rope_parameters(
    hidden_size: int,
    num_attention_heads: int,
    device: Optional["torch.device"] = None,
    rope_theta: float = 10000.0,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    Args:
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    factor = 8.0

    # Gets the default RoPE parameters
    base = rope_theta
    partial_rotary_factor = 1.0
    head_dim = hidden_size // num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor
    return inv_freq, attention_factor


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
    ):
        super().__init__()
        self.rope_kwargs = {
            "rope_type": rope_type,
            "factor": scaling_factor,
            "dim": hidden_size,
            "base": base,
            "max_position_embeddings": max_position_embeddings,
        }
        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings

        inv_freq, self.attention_scaling = compute_linear_scaling_rope_parameters(hidden_size, num_attention_heads, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QMLP(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SdpaAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 attention_dropout: float = 0.15,
                 use_qk_norm: bool = True,
                 elementwise_attn_output_gate: bool = True,
                 rms_norm_eps: float = 1e-6
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        if self.elementwise_attn_output_gate:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor
    ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.elementwise_attn_output_gate:
            query_states = query_states.view(bsz, q_len, self.num_key_value_heads, -1)
            query_states, gate_score = torch.split(
                query_states, [self.head_dim * self.num_key_value_groups, self.head_dim * self.num_key_value_groups], dim=-1)
            gate_score = gate_score.reshape(bsz, q_len, -1, self.head_dim)
            query_states = query_states.reshape(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # key_states: bs, head, q_len, head_dim
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        if self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class TransformerEncoder(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 attention_dropout: float = 0.15,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.mlp = QMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = SdpaAttention(hidden_size,
                                       num_attention_heads,
                                       num_key_value_heads,
                                       attention_dropout, use_qk_norm=True,
                                       elementwise_attn_output_gate=True,
                                       rms_norm_eps=rms_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,
                position_embeddings: torch.Tensor
                ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CompleteBinBaseModel(nn.Module):

    def __init__(self,
                 kmer_dim,
                 classes_num,
                 split_parts_list: List,
                 dropout: float,
                 rotary_apply,
                 hidden_dim: int = 768,
                 layers=4,
                 output_embedding=False) -> None:
        super(CompleteBinBaseModel, self).__init__()
        self.classes_num = classes_num
        self.rotary_apply = rotary_apply
        logger.info(f"--> hidden dim: {hidden_dim}, layers: {layers}. Fix")
        self.layers = layers
        self.output_embedding = output_embedding
        self.seq_length = sum(split_parts_list)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, sum(split_parts_list) + 100, hidden_dim, requires_grad=True),
            requires_grad=True)
        # seq
        self.rotary = RotaryEmbedding(hidden_dim, 8, max_position_embeddings=sum(split_parts_list))
        self.token_proj = nn.Linear(kmer_dim, hidden_dim, bias=False)
        self.transformer_model = nn.ModuleList(
            [TransformerEncoder(hidden_dim, hidden_dim * 2, 8, 8, dropout)
             for _ in range(self.layers)]
        )
        if classes_num == 0 or classes_num is None:
            self.out_linear = None
        else:
            self.out_linear = nn.Linear(hidden_dim, classes_num, bias=False)

    def get_token_proj(self, x):
        x = self.token_proj(x)
        return x

    def get_feature_of_tokens(self, x):
        _, l, _ = x.shape
        x += self.pos_embedding[:, 0: l]
        if self.rotary_apply:
            position_ids = torch.arange(0, self.seq_length).unsqueeze(0).to(x.device)
            position_embed = self.rotary(x, position_ids)
        else:
            position_embed = None
        for i in range(self.layers):
            x = self.transformer_model[i](x, position_embed)
        return x

    def forward(self, seq_tokens_inputs):
        seq_fea_enc = encode_seq2vec(self.get_feature_of_tokens(self.get_token_proj(seq_tokens_inputs)))
        if self.output_embedding or self.out_linear is None:
            return seq_fea_enc
        return self.out_linear(seq_fea_enc)
