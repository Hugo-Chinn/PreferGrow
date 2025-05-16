import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#from einops import rearrange
#from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
# from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

# from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            #nn.SiLU(),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################
class RotaryEmbedding(nn.Module):
    def __init__(self, seq_len, dim, base=10000):
        super().__init__()
        self.dim = dim  # 需要与hidden_dim匹配
        self.base = base
        
        self.thetas = 1.0 / ( base ** (torch.arange(0, dim, 2).float() / dim))
        
        self.cos_cached = None
        self.sin_cached = None
        self._build_cos_sin(seq_len)

    def _build_cos_sin(self, seq_len):
        t = torch.arange(seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.thetas)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.cos_cached = emb.cos().unsqueeze(0)
        self.sin_cached = emb.sin().unsqueeze(0)
    
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        return (x * self.cos_cached.to(x.device)) + (self.rotate_half(x) * self.sin_cached.to(x.device))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]
        
        :return: A 3d tensor with shape of (N, T_q, C)
        
        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection
        output_res = output + queries
        
        return output_res

# class DDiTBlock_UCUR(nn.Module):

#     def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
#         super().__init__()
#         self.n_heads = n_heads

#         self.norm1 = LayerNorm(dim)
#         self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
#         self.attn_out = nn.Linear(dim, dim, bias=False)
#         self.dropout1 = nn.Dropout(dropout)

#         self.norm2 = LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_ratio * dim, bias=True),
#             nn.GELU(approximate="tanh"),
#             nn.Linear(mlp_ratio * dim, dim, bias=True)
#         )
#         self.dropout2 = nn.Dropout(dropout)

#         self.dropout = dropout
        

#         #self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
#         #self.adaLN_modulation.weight.data.zero_()
#         #self.adaLN_modulation.bias.data.zero_()


#     def _get_bias_dropout_scale(self):
#         return (
#             bias_dropout_add_scale_fused_train
#             if self.training
#             else bias_dropout_add_scale_fused_inference
#         )


#     def forward(self, x, rotary_cos_sin, seqlens=None, interleaved=True):
#         batch_size, seq_len = x.shape[0], x.shape[1]

#         #bias_dropout_scale_fn = self._get_bias_dropout_scale()

#         #shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

#         # attention operation
#         x_skip = x
#         #x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
#         x = self.norm1(x)
#         # dtype0 = x.dtype

#         qkv = self.attn_qkv(x)
#         qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
#         with torch.cuda.amp.autocast(enabled=False):
#             cos, sin = rotary_cos_sin
#             interleaved = torch.tensor(interleaved, dtype=torch.bool)
#             qkv = rotary.apply_rotary_pos_emb(
#                 qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), interleaved
#             )
#         qkv = rearrange(qkv, 'b s ... -> (b s) ...')
#         if seqlens is None:
#             cu_seqlens = torch.arange(
#                 0, (batch_size + 1) * seq_len, step=seq_len,
#                 dtype=torch.int32, device=qkv.device
#             )
#         else:
#             cu_seqlens = seqlens.cumsum(-1)
#         x = flash_attn_varlen_qkvpacked_func(
#             qkv, cu_seqlens, seq_len, 0., causal=False)
        
#         x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

#         #x = bias_dropout_scale_fn(self.attn_out(x), None, None, None, self.dropout)
#         x = F.dropout(self.attn_out(x), p=self.dropout, training=self.training)
#         # res: x_skip

#         # mlp operation
#         #x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, None, None, self.dropout)
#         x = F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training)
#         # res: x
#         return x

class DDiTBlock_UCUR(nn.Module):

    def __init__(self, dim, n_heads, seq_len, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.position_embedding = RotaryEmbedding(seq_len, dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, dim, n_heads, dropout)
        self.norm2 = LayerNorm(dim)
        self. mlp = PositionwiseFeedForward(dim, dim, dropout)
        self.norm3 = LayerNorm(dim)

    def forward(self, x):
        #batch_size, seq_len = x.shape[0], x.shape[1]
        inputs_emb = self.position_embedding(x)
        inputs_emb = self.emb_dropout(inputs_emb)
        #mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        #seq *= mask
        seq_normalized = self.norm1(inputs_emb)
        mh_attn_out = self.attn(seq_normalized, inputs_emb)
        ff_out = self.mlp(self.norm2(mh_attn_out))
        #ff_out *= mask
        ff_out = self.norm3(ff_out)
        #state_hidden = extract_axis_1(ff_out, len_states - 1)
        return ff_out

"""
class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)
        
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x
"""


class ParameterLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]
    
class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_dim+1, embedding_dim=dim, padding_idx=vocab_dim)
        #torch.nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        torch.nn.init.normal_(self.embedding.weight, 0, 1)

    def forward(self, x):
        return self.embedding(x)
    
    def weight(self,):
        return self.embedding.weight

# class DDitFinalLayer4Rec(nn.Module):
#     def __init__(self, hidden_size, out_channels, cond_dim):
#         super().__init__()
#         self.norm_final = LayerNorm(hidden_size)
#         self.linear = nn.Linear(hidden_size, out_channels)
#         self.linear.weight.data.zero_()
#         self.linear.bias.data.zero_()

#         self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
#         self.adaLN_modulation.weight.data.zero_()
#         self.adaLN_modulation.bias.data.zero_()


#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
#         x = modulate_fused(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x

# class DDitFinalLayer(nn.Module):
#     def __init__(self, hidden_size, out_channels, cond_dim):
#         super().__init__()
#         self.norm_final = LayerNorm(hidden_size)
#         self.linear = nn.Linear(hidden_size, out_channels)
#         self.linear.weight.data.zero_()
#         self.linear.bias.data.zero_()

#         self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
#         self.adaLN_modulation.weight.data.zero_()
#         self.adaLN_modulation.bias.data.zero_()


#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
#         x = modulate_fused(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x

# class DDitFinalLayer4Rec(nn.Module):
#     def __init__(self, hidden_size, out_channels, cond_dim):
#         super().__init__()
#         self.linear = nn.Linear(cond_dim + hidden_size, hidden_size)
#         self.act = nn.GELU()
#         self.s_fc = nn.Linear(hidden_size, out_channels)
#         #self.norm_final = LayerNorm(hidden_size)


#     def forward(self, x, c):
#         x_in = torch.cat((x, c.unsqueeze(1)), dim=2)
#         x = self.linear(x_in)
#         x = self.act(x)
#         x_out = self.s_fc(x)
#         #shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
#         #x = modulate_fused(self.norm_final(x), shift, scale)
#         #x = self.linear(x)
#         return x_out

class DDitFinalLayer4Rec(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim + hidden_size, out_channels)

    def forward(self, x, c):
        x_in = torch.cat((x, c.unsqueeze(1)), dim=2)
        x_out = self.linear(x_in)
        return x_out

# class DDitFinalLayer4Rec(nn.Module):
#     def __init__(self, hidden_size, out_channels, cond_dim):
#         super().__init__()
#         self.linear = nn.Linear(cond_dim, hidden_size)

#     def forward(self, x):
#         x_out = self.linear(x)
#         return x_out



# class SEDD4REC(nn.Module, PyTorchModelHubMixin):
#     def __init__(self, config):
#         super().__init__()

#         # hack to make loading in configs easier
#         if type(config) == dict:
#             config = OmegaConf.create(config)

#         self.config = config

#         self.absorb = config.graph.type == "absorb"
#         if config.training.data == "ATV":
#             item_num = config.data.ATV.item_num
#         #seq_len = config.data.ATV.seq_len
#         #item_num = config.data.ATV.item_num
#         elif config.training.data == "ML1M":
#             item_num = config.data.ML1M.item_num
#         vocab_size = item_num + (1 if self.absorb else 0)
#         print(vocab_size)

#         self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
#         self.sigma_map = TimestepEmbedder(config.model.cond_dim)
#         self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

#         self.history_blocks = nn.ModuleList([
#             DDiTBlock_UCUR(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
#         ])

#         self.output_layer = DDitFinalLayer4Rec(config.model.hidden_size, vocab_size, config.model.cond_dim + config.model.hidden_size)
#         self.scale_by_sigma = config.model.scale_by_sigma

    
#     def _get_bias_dropout_scale(self):
#         return (
#             bias_dropout_add_scale_fused_train
#             if self.training
#             else bias_dropout_add_scale_fused_inference
#         )


#     def forward(self, history_indices, noisy_indices, sigma):

#         h = self.vocab_embed(history_indices) # B×L→B×L×d
#         x = self.vocab_embed(noisy_indices)  # B→B×d
#         # noise level condition
#         c_t = F.silu(self.sigma_map(sigma))   # B→B×d_c
#         # history condition
#         rotary_cos_sin = self.rotary_emb(h)
#         with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#             for i in range(len(self.history_blocks)):
#                 h = self.history_blocks[i](h, rotary_cos_sin, seqlens=None, interleaved=True) # B×L×d→B×L×d
#             c_h = h[:,-1,:] # B×L×d→B×d
#         # mix condition c = c_h + c_t
#         #print(c_h.size())
#         #print(c_t.size())
#         c = torch.cat((c_h, c_t), dim=-1) # B×d+B×d_c→B×(d+d_c) 
#         x = self.output_layer(x,c) # concrete score

#         if self.scale_by_sigma:
#             assert self.absorb, "Haven't configured this to work."
#             esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
#             x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
            
#         x = torch.scatter(x, -1, noisy_indices[..., None], torch.zeros_like(x[..., :1]))

#         return x

#PyTorchModelHubMixin
class SEDD4REC(nn.Module):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        self.is_disliked_item = config.graph.is_disliked_item
        if config.training.data == "ATV":
            item_num = config.data.ATV.item_num
            seq_len = config.data.ATV.seq_len
        #item_num = config.data.ATV.item_num
        elif config.training.data == "ML1M":
            item_num = config.data.ML1M.item_num
            seq_len = config.data.ML1M.seq_len
        elif config.training.data == "ATG":
            item_num = config.data.ATG.item_num
            seq_len = config.data.ATG.seq_len
        #item_num = config.data.ATV.item_num
        elif config.training.data == "ASO":
            item_num = config.data.ASO.item_num
            seq_len = config.data.ASO.seq_len
        elif config.training.data == "Steam":
            item_num = config.data.Steam.item_num
            seq_len = config.data.Steam.seq_len
        #item_num = config.data.ATV.item_num
        elif config.training.data == "Beauty":
            item_num = config.data.Beauty.item_num
            seq_len = config.data.Beauty.seq_len
        vocab_size = item_num + (1 if self.is_disliked_item else 0)
        print(self.is_disliked_item)
        print(vocab_size)
        self.score_flag = config.model.score_flag
        self.score_method = config.model.score_method

        self.nonpreference_user_ratio = config.training.nonpreference_user_ratio

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        # self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.nonpreference_user = nn.Embedding(
            num_embeddings=1,
            embedding_dim=config.model.hidden_size,
        )
        nn.init.normal_(self.nonpreference_user.weight, 0, 1)

        #self.none_user = 

        # self.history_blocks = nn.ModuleList([
        #     DDiTBlock_UCUR(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        # ])
        self.history_blocks = nn.ModuleList([
            DDiTBlock_UCUR(config.model.hidden_size, config.model.n_heads, seq_len, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        ])
        if self.score_flag:
            self.output_layer = DDitFinalLayer4Rec(config.model.hidden_size, config.model.hidden_size, config.model.cond_dim + config.model.hidden_size)
        else:
            self.output_layer = DDitFinalLayer4Rec(config.model.hidden_size, vocab_size, config.model.cond_dim + config.model.hidden_size)
        self.scale_by_sigma = config.model.scale_by_sigma

    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, history_indices, noisy_indices, sigma):

        h = self.vocab_embed(history_indices) # B×L→B×L×d
        x = self.vocab_embed(noisy_indices)  # B→B×d
        # noise level condition
        c_t = F.silu(self.sigma_map(sigma))   # B→B×d_c
        # history condition
        # rotary_cos_sin = self.rotary_emb(h)
        for i in range(len(self.history_blocks)):
            #h = self.history_blocks[i](h, rotary_cos_sin, seqlens=None, interleaved=True) # B×L×d→B×L×d
            h = self.history_blocks[i](h) # B×L×d→B×L×d
        c_h = h[:,-1,:] # B×L×d→B×d

        if self.nonpreference_user_ratio > 0:
            B, D = c_h.shape
            mask = (torch.rand(B, 1, device=c_h.device) > self.nonpreference_user_ratio).float()
            c_h = c_h * mask + self.nonpreference_user(torch.zeros(1, dtype=torch.long, device=c_h.device)) * (1 - mask)

        c = torch.cat((c_h, c_t), dim=-1) # B×d+B×d_c→B×(d+d_c) 
        x_u = self.output_layer(x, c) # concrete score
        score = self.cacu_score(x_u, x)

        # if self.scale_by_sigma:
        #     assert self.absorb, "Haven't configured this to work."
        #     esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
        #     x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
        score = torch.scatter(score, -1, noisy_indices[..., None], torch.zeros_like(score[..., :1]))
        return score
    
    def cacu_score(self, x_u, x_t):
        if self.score_flag :
            item_embs = self.vocab_embed.weight()[:-1]
            if self.score_method == "diffcos":
                score = torch.matmul(x_u - x_t, item_embs.T)
            elif self.score_method == "diffcosNorm":
                diff = F.normalize(x_u - x_t, dim=-1)
                item_embs_norm = F.normalize(item_embs, dim=-1)
                score = torch.matmul(diff, item_embs_norm.T)
            elif self.score_method == "oricos":
                score = torch.matmul(x_u, item_embs.T)
            elif self.score_method == "oricosNorm1":
                diff = F.normalize(x_u, dim=-1)
                item_embs_norm = F.normalize(item_embs, dim=-1)
                score = torch.matmul(diff, item_embs_norm.T)
            elif self.score_method == "oricosNorm2":
                diff = F.normalize(x_u, dim=-1)
                score = torch.matmul(diff, item_embs.T)
        else:
            score = x_u
        return score

    @torch.no_grad()
    def forward_eval(self, history_indices, noisy_indices, sigma, personalized_strength):

        h = self.vocab_embed(history_indices) # B×L→B×L×d
        x = self.vocab_embed(noisy_indices)  # B→B×d
        # noise level condition
        c_t = F.silu(self.sigma_map(sigma))   # B→B×d_c

        #print(noisy_indices.shape)
        # history condition
        # rotary_cos_sin = self.rotary_emb(h)
        for i in range(len(self.history_blocks)):
            #h = self.history_blocks[i](h, rotary_cos_sin, seqlens=None, interleaved=True) # B×L×d→B×L×d
            h = self.history_blocks[i](h) # B×L×d→B×L×d
        c_h = h[:,-1,:] # B×L×d→B×d
        c_ht = torch.cat((c_h, c_t), dim=-1) # B×d+B×d_c→B×(d+d_c) 

        nonprefer_u = self.nonpreference_user(torch.tensor([0]).to(c_h.device))
        nonprefer_u = nonprefer_u.repeat(noisy_indices.shape[0], 1)
        c_ut = torch.cat((nonprefer_u, c_t), dim=-1) # B×d+B×d_c→B×(d+d_c)

        x_ht = self.output_layer(x, c_ht)
        score_ht = self.cacu_score(x_ht, x)
        x_ut = self.output_layer(x, c_ut)
        score_ut = self.cacu_score(x_ut, x)

        personalized_prefer_ranks = personalized_strength * score_ht + \
                       (1 - personalized_strength) * score_ut

        personalized_prefer_ranks = torch.scatter(personalized_prefer_ranks, -1, noisy_indices[..., None], torch.zeros_like(personalized_prefer_ranks[..., :1]))
        return personalized_prefer_ranks
    
# class SEDD(nn.Module, PyTorchModelHubMixin):
#     def __init__(self, config):
#         super().__init__()

#         # hack to make loading in configs easier
#         if type(config) == dict:
#             config = OmegaConf.create(config)

#         self.config = config

#         self.absorb = config.graph.type == "absorb"
#         vocab_size = config.tokens + (1 if self.absorb else 0)

#         self.vocab_embed = ParameterLayer(config.model.hidden_size, vocab_size)
#         self.sigma_map = TimestepEmbedder(config.model.cond_dim)
#         self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

#         self.blocks = nn.ModuleList([
#             DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
#         ])

#         self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
#         self.scale_by_sigma = config.model.scale_by_sigma

    
#     def _get_bias_dropout_scale(self):
#         return (
#             bias_dropout_add_scale_fused_train
#             if self.training
#             else bias_dropout_add_scale_fused_inference
#         )


#     def forward(self, indices, sigma):

#         x = self.vocab_embed(indices)
#         c = F.silu(self.sigma_map(sigma))

#         rotary_cos_sin = self.rotary_emb(x)

#         with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#             for i in range(len(self.blocks)):
#                 x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
 
#             x = self.output_layer(x, c)


#         if self.scale_by_sigma:
#             assert self.absorb, "Haven't configured this to work."
#             esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
#             x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
            
#         x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

#         return x

