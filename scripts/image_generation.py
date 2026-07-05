import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌀 Diffusion Trajectory — Latent Space Explorer (UMAP-3D)

    Welcome to the Diffusion Trajectory Explorer. This notebook visualizes the step-by-step denoising process of a state-of-the-art **Just-image-Transformer (JiT)** model in a non-linear 3D space.

    By projecting the high-dimensional image generation process (256x256x3 = 196,608 dimensions) down to 3 dimensions using **UMAP**, we can observe how the neural network conceptually "navigates" from pure random noise to a perfectly coherent image.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import math
    from math import pi
    import torch
    from torch import nn
    import torch.nn.functional as F
    import numpy as np
    from einops import rearrange, repeat
    import umap
    from PIL import Image
    import plotly.graph_objects as go
    import base64
    from io import BytesIO
    import os

    # Try importing huggingface_hub for model downloading
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
    return (
        BytesIO,
        F,
        Image,
        base64,
        go,
        hf_hub_download,
        math,
        mo,
        nn,
        np,
        pi,
        rearrange,
        repeat,
        torch,
        umap,
    )


@app.cell
def _(nn, np, pi, rearrange, repeat, torch):
    # --- model_util.py ---
    def broadcat(tensors, dim=-1):
        num_tensors = len(tensors)
        shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
        assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
        shape_len = list(shape_lens)[0]
        dim = (dim + shape_len) if dim < 0 else dim
        dims = list(zip(*map(lambda t: list(t.shape), tensors)))
        expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
        assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
        max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
        expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
        expanded_dims.insert(dim, (dim, dims[dim]))
        expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
        tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
        return torch.cat(tensors, dim=dim)

    def rotate_half(x):
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')

    class VisionRotaryEmbedding(nn.Module):
        def __init__(self, dim, pt_seq_len, ft_seq_len=None, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1):
            super().__init__()
            if custom_freqs:
                freqs = custom_freqs
            elif freqs_for == 'lang':
                freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
            elif freqs_for == 'pixel':
                freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
            elif freqs_for == 'constant':
                freqs = torch.ones(num_freqs).float()
            else:
                raise ValueError(f'unknown modality {freqs_for}')

            if ft_seq_len is None: ft_seq_len = pt_seq_len
            t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

            freqs_h = torch.einsum('..., f -> ... f', t, freqs)
            freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)

            freqs_w = torch.einsum('..., f -> ... f', t, freqs)
            freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)

            freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

            self.register_buffer("freqs_cos", freqs.cos())
            self.register_buffer("freqs_sin", freqs.sin())

        def forward(self, t, start_index=0):
            rot_dim = self.freqs_cos.shape[-1]
            end_index = start_index + rot_dim
            t_left, t_mid, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
            t_mid = (t_mid * self.freqs_cos) + (rotate_half(t_mid) * self.freqs_sin)
            return torch.cat((t_left, t_mid, t_right), dim=-1)

    class VisionRotaryEmbeddingFast(nn.Module):
        def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1, num_cls_token=0):
            super().__init__()
            if custom_freqs:
                freqs = custom_freqs
            elif freqs_for == 'lang':
                freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
            elif freqs_for == 'pixel':
                freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
            elif freqs_for == 'constant':
                freqs = torch.ones(num_freqs).float()
            else:
                raise ValueError(f'unknown modality {freqs_for}')

            if ft_seq_len is None: ft_seq_len = pt_seq_len
            t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

            freqs = torch.einsum('..., f -> ... f', t, freqs)
            freqs = repeat(freqs, '... n -> ... (n r)', r=2)
            freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

            if num_cls_token > 0:
                freqs_flat = freqs.view(-1, freqs.shape[-1])
                cos_img = freqs_flat.cos()
                sin_img = freqs_flat.sin()

                N_img, D = cos_img.shape
                cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
                sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

                self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()
                self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
            else:
                self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
                self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

        def forward(self, t): 
            return t * self.freqs_cos + rotate_half(t) * self.freqs_sin

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
            return (self.weight * hidden_states).to(input_dtype)

    def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    return RMSNorm, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


@app.cell
def _(
    F,
    RMSNorm,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
    math,
    nn,
    torch,
):
    # --- model_jit.py ---
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    class BottleneckPatchEmbed(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
            super().__init__()
            img_size = (img_size, img_size)
            patch_size = (patch_size, patch_size)
            num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
            self.img_size = img_size
            self.patch_size = patch_size
            self.num_patches = num_patches

            self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
            self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

        def forward(self, x):
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1]
            x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
            return x

    class TimestepEmbedder(nn.Module):
        def __init__(self, hidden_size, frequency_embedding_size=256):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
            self.frequency_embedding_size = frequency_embedding_size

        @staticmethod
        def timestep_embedding(t, dim, max_period=10000):
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
        def __init__(self, num_classes, hidden_size):
            super().__init__()
            self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
            self.num_classes = num_classes

        def forward(self, labels):
            embeddings = self.embedding_table(labels)
            return embeddings

    def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

        with torch.cuda.amp.autocast(enabled=False):
            attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    class Attention(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads

            self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
            self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x, rope):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = self.q_norm(q)
            k = self.k_norm(k)
            q = rope(q)
            k = rope(k)

            x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
            x = x.transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class SwiGLUFFN(nn.Module):
        def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
            super().__init__()
            hidden_dim = int(hidden_dim * 2 / 3)
            self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
            self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
            self.ffn_dropout = nn.Dropout(drop)

        def forward(self, x):
            x12 = self.w12(x)
            x1, x2 = x12.chunk(2, dim=-1)
            hidden = F.silu(x1) * x2
            return self.w3(self.ffn_dropout(hidden))

    class FinalLayer(nn.Module):
        def __init__(self, hidden_size, patch_size, out_channels):
            super().__init__()
            self.norm_final = RMSNorm(hidden_size)
            self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

        @torch.compile
        def forward(self, x, c):
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
            return x

    class JiTBlock(nn.Module):
        def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
            super().__init__()
            self.norm1 = RMSNorm(hidden_size, eps=1e-6)
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                  attn_drop=attn_drop, proj_drop=proj_drop)
            self.norm2 = RMSNorm(hidden_size, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

        @torch.compile
        def forward(self, x,  c, feat_rope=None):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x

    class JiT(nn.Module):
        def __init__(self, input_size=256, patch_size=16, in_channels=3, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, num_classes=1000, bottleneck_dim=128, in_context_len=32, in_context_start=8):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = in_channels
            self.patch_size = patch_size
            self.num_heads = num_heads
            self.hidden_size = hidden_size
            self.input_size = input_size
            self.in_context_len = in_context_len
            self.in_context_start = in_context_start
            self.num_classes = num_classes

            self.t_embedder = TimestepEmbedder(hidden_size)
            self.y_embedder = LabelEmbedder(num_classes, hidden_size)
            self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

            if self.in_context_len > 0:
                self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
                torch.nn.init.normal_(self.in_context_posemb, std=.02)

            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
            self.feat_rope_incontext = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len)

            self.blocks = nn.ModuleList([
                JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                         attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                         proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
                for i in range(depth)
            ])

            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
            self.initialize_weights()

        def initialize_weights(self):
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)

            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

        def unpatchify(self, x, p):
            c = self.out_channels
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
            return imgs

        def forward(self, x, t, y):
            t_emb = self.t_embedder(t)
            y_emb = self.y_embedder(y)
            c = t_emb + y_emb

            x = self.x_embedder(x)
            x += self.pos_embed

            for i, block in enumerate(self.blocks):
                if self.in_context_len > 0 and i == self.in_context_start:
                    in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                    in_context_tokens += self.in_context_posemb
                    x = torch.cat([in_context_tokens, x], dim=1)
                x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

            x = x[:, self.in_context_len:]
            x = self.final_layer(x, c)
            output = self.unpatchify(x, self.patch_size)
            return output

    def JiT_B_16(**kwargs): return JiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)
    def JiT_B_32(**kwargs): return JiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)
    def JiT_L_16(**kwargs): return JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)
    def JiT_L_32(**kwargs): return JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)
    def JiT_H_16(**kwargs): return JiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)
    def JiT_H_32(**kwargs): return JiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)

    JiT_models = {
        'JiT-B/16': JiT_B_16, 'JiT-B/32': JiT_B_32,
        'JiT-L/16': JiT_L_16, 'JiT-L/32': JiT_L_32,
        'JiT-H/16': JiT_H_16, 'JiT-H/32': JiT_H_32,
    }
    return (JiT_models,)


@app.cell
def _(JiT_models, nn, torch):
    # --- denoiser.py ---
    class Denoiser(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.net = JiT_models[args.model](
                input_size=args.img_size,
                in_channels=3,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
            )
            self.img_size = args.img_size
            self.num_classes = args.class_num

            self.label_drop_prob = args.label_drop_prob
            self.P_mean = args.P_mean
            self.P_std = args.P_std
            self.t_eps = args.t_eps
            self.noise_scale = args.noise_scale

            self.ema_decay1 = args.ema_decay1
            self.ema_decay2 = args.ema_decay2
            self.ema_params1 = None
            self.ema_params2 = None

            self.method = args.sampling_method
            self.steps = args.num_sampling_steps
            self.cfg_scale = args.cfg
            self.cfg_interval = (args.interval_min, args.interval_max)

        def drop_labels(self, labels):
            drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
            return out

        def sample_t(self, n: int, device=None):
            z = torch.randn(n, device=device) * self.P_std + self.P_mean
            return torch.sigmoid(z)

        def forward(self, x, labels):
            labels_dropped = self.drop_labels(labels) if self.training else labels

            t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
            e = torch.randn_like(x) * self.noise_scale

            z = t * x + (1 - t) * e
            v = (x - z) / (1 - t).clamp_min(self.t_eps)

            x_pred = self.net(z, t.flatten(), labels_dropped)
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

            loss = (v - v_pred) ** 2
            loss = loss.mean(dim=(1, 2, 3)).mean()

            return loss

        @torch.no_grad()
        def generate(self, labels):
            device = labels.device
            bsz = labels.size(0)
            z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
            timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

            if self.method == "euler": stepper = self._euler_step
            elif self.method == "heun": stepper = self._heun_step
            else: raise NotImplementedError

            for i in range(self.steps - 1):
                t = timesteps[i]
                t_next = timesteps[i + 1]
                z = stepper(z, t, t_next, labels)
            z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
            return z

        @torch.no_grad()
        def _forward_sample(self, z, t, labels):
            x_cond = self.net(z, t.flatten(), labels)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

            x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

            low, high = self.cfg_interval
            interval_mask = (t < high) & ((low == 0) | (t > low))
            cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

            return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

        @torch.no_grad()
        def _euler_step(self, z, t, t_next, labels):
            v_pred = self._forward_sample(z, t, labels)
            z_next = z + (t_next - t) * v_pred
            return z_next

        @torch.no_grad()
        def _heun_step(self, z, t, t_next, labels):
            v_pred_t = self._forward_sample(z, t, labels)
            z_next_euler = z + (t_next - t) * v_pred_t
            v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

            v_pred = 0.5 * (v_pred_t + v_pred_t_next)
            z_next = z + (t_next - t) * v_pred
            return z_next

    return (Denoiser,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📖 Understanding the Process

    Diffusion models generate images by starting with random noise and progressively denoising it. This trajectory can be broken down into three intuitive phases:

    1. **Chaos (🔴)**: At the beginning, the noise level $\sigma$ is extremely high. The model makes large, sweeping changes to establish the global layout and color palette.
    2. **Exploration (🟡)**: The broad structure is set, and the model starts "committing" to specific shapes and textures. Velocity in the latent space drops as the model refines its initial guesses.
    3. **Convergence (🟢)**: The noise is almost gone. The model only makes minute, high-frequency corrections (like sharpening edges). In the UMAP projection, the points are very close to the final destination.

    > *Note: UMAP is a non-linear dimensionality reduction technique. It preserves local relationships well but distorts global distances. The "velocity" you see plotted is the distance traveled in the 3D UMAP space, not the raw pixel space.*
    """)
    return


@app.cell
def _(Denoiser, Image, hf_hub_download, mo, np, torch, umap):
    CLASSES = {
        207:  "Golden Retriever",
        980:  "Volcano",
        1:    "Goldfish",
        388:  "Giant Panda",
        985:  "Daisy",
    }
    THUMB_EVERY = 5
    THUMB_SIZE  = 64
    METHOD      = "UMAP-3D"

    class Args:
        model          = 'JiT-B/16'
        img_size       = 256
        class_num      = 1000
        attn_dropout   = 0.0
        proj_dropout   = 0.0
        label_drop_prob= 0.1
        P_mean         = -0.8
        P_std          = 0.8
        t_eps          = 1e-5
        noise_scale    = 1.0
        ema_decay1     = 0.999
        ema_decay2     = 0.9999
        sampling_method= 'euler'
        num_sampling_steps = 250
        cfg            = 5.0
        interval_min   = 0.1
        interval_max   = 1.0

    mo.stop(
        hf_hub_download is None, 
        mo.md("⚠️ **Error:** `huggingface_hub` is not installed. Please run `pip install huggingface_hub` to download the model automatically.")
    )

    with mo.status.spinner("Fetching model & Generating trajectories (This may take a few minutes)...") as _spinner:
        torch.set_float32_matmul_precision('high')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        args = Args()
        denoiser = Denoiser(args).to(device)

        _spinner.update("Downloading weights from HuggingFace...")
        model_path = hf_hub_download(repo_id="avonne/Just-image-Transformer", filename="jit-b-16/checkpoint-last.pth")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        cleaned = {k[4:] if k.startswith("net.") else k: v for k, v in state_dict.items()}
        denoiser.net.load_state_dict(cleaned, strict=True)
        denoiser.eval()

        all_frames_flat = {}
        all_thumbs      = {}
        all_sigmas      = {}
        N_steps = 0

        # Sequentially generate classes to prevent CUDA OOM on smaller VRAM GPUs
        for class_id, class_name in CLASSES.items():
            _spinner.update(f"Generating trajectory for {class_name}...")

            historique = []
            def hook(module, args_in):
                # Save just the single batch item to CPU memory to avoid VRAM hoarding
                historique.append(args_in[0][0:1].detach().clone().cpu())

            handle = denoiser.net.register_forward_pre_hook(hook)

            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)

            labels = torch.tensor([class_id], device=device)
            with torch.no_grad():
                final = denoiser.generate(labels)

            handle.remove()
            historique.append(final[0:1].cpu())

            if N_steps == 0:
                N_steps = len(historique)

            # Flatten for UMAP
            flat = np.stack([h.flatten().numpy() for h in historique])
            all_frames_flat[class_id] = flat

            # Extract thumbnails
            thumbs = []
            for t in historique[::THUMB_EVERY]:
                arr = t[0].numpy().transpose(1, 2, 0)
                arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr).resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR)
                thumbs.append(np.array(img))
            all_thumbs[class_id] = np.stack(thumbs)

            # Sigmas
            t_vals = np.linspace(0, 1, N_steps)
            all_sigmas[class_id] = np.exp(-2.5 * t_vals**2)

            # Clear the VRAM aggressively after each class
            torch.cuda.empty_cache()

        _spinner.update("Applying UMAP (3D) dimensionality reduction...")
        all_flat_concat = np.concatenate(list(all_frames_flat.values()), axis=0)
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        all_coords_3d = reducer.fit_transform(all_flat_concat)

        start_idx = 0
        data = {}
        for class_id in CLASSES:
            n_steps_class = all_frames_flat[class_id].shape[0]
            coords_3d = all_coords_3d[start_idx:start_idx + n_steps_class]
            start_idx += n_steps_class
            data[class_id] = (coords_3d, all_thumbs[class_id], all_sigmas[class_id])

    CLASS_COLORS = {
        207:  ("#6c63f0", "#a78bfa", "#f0abfc"),
        980:  ("#ef4444", "#f97316", "#fbbf24"),
        1:    ("#06b6d4", "#22d3ee", "#a5f3fc"),
        388:  ("#22c55e", "#4ade80", "#bbf7d0"),
        985:  ("#f472b6", "#fb7185", "#fda4af"),
    }
    N = N_steps
    return CLASSES, CLASS_COLORS, METHOD, N, THUMB_EVERY, data


@app.cell
def _(BytesIO, CLASSES, Image, base64, data, np):
    def compute_speeds(coords):
        return np.linalg.norm(np.diff(coords, axis=0), axis=1)

    speeds_all  = {cid: compute_speeds(data[cid][0]) for cid in CLASSES}
    percentiles = {
        cid: (np.percentile(s, 33), np.percentile(s, 66))
        for cid, s in speeds_all.items()
    }

    def get_phase(step, class_id):
        s    = speeds_all[class_id][min(step, len(speeds_all[class_id]) - 1)]
        p33, p66 = percentiles[class_id]
        if s > p66:   return "Chaos",       "#ef4444"
        elif s > p33: return "Exploration", "#fbbf24"
        else:         return "Convergence", "#4ade80"

    def arr_to_b64(arr):
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    thumbs_b64 = {}
    for _cid in CLASSES:
        _, _thumbs, _ = data[_cid]
        thumbs_b64[_cid] = [arr_to_b64(_thumbs[i]) for i in range(_thumbs.shape[0])]
    return get_phase, speeds_all, thumbs_b64


@app.cell
def _(N, mo):
    get_step, set_step = mo.state(0)

    step_slider = mo.ui.slider(
        0, N - 1, 
        value=get_step(),       # Relie le slider à l'état
        on_change=set_step,     # Met à jour l'état quand le slider bouge
        label="Denoising step",
        full_width=True,
    )

    controls = mo.vstack([step_slider], gap="0.25rem")
    return controls, set_step, step_slider


@app.cell
def _(CLASSES, mo):
    class_checkboxes = mo.ui.multiselect(
        options=list(CLASSES.values()),
        value=list(CLASSES.values()),
        label="Visible classes",
    )

    show_ghost  = mo.ui.checkbox(value=True,  label="Full trajectory (ghost)")
    show_sphere = mo.ui.checkbox(value=True,  label="Noise sphere σ (approximate)")
    show_vector = mo.ui.checkbox(value=True,  label="Displacement vector")
    show_final  = mo.ui.checkbox(value=True,  label="Final point ◆")

    options_row = mo.vstack([
        class_checkboxes,
        mo.vstack([show_ghost, show_sphere, show_vector, show_final], gap="0.25rem"),
    ], gap="1rem")
    return (
        class_checkboxes,
        options_row,
        show_final,
        show_ghost,
        show_sphere,
        show_vector,
    )


@app.cell
def _(
    CLASSES,
    CLASS_COLORS,
    METHOD,
    N,
    class_checkboxes,
    controls,
    data,
    go,
    mo,
    np,
    options_row,
    show_final,
    show_ghost,
    show_sphere,
    show_vector,
    step_slider,
):
    step        = step_slider.value
    visible_ids = [cid for cid, name in CLASSES.items() if name in class_checkboxes.value]

    fig = go.Figure()

    for _class_id in CLASSES:
        if _class_id not in visible_ids:
            continue

        _coords, _thumbs, _sigmas = data[_class_id]
        _c0, _c1, _c2 = CLASS_COLORS[_class_id]
        _name      = CLASSES[_class_id]
        _sigma_cur = float(_sigmas[step])

        if show_ghost.value:
            fig.add_trace(go.Scatter3d(
                x=_coords[:, 0], y=_coords[:, 1], z=_coords[:, 2],
                mode="lines",
                line=dict(color=_c0, width=1),
                opacity=0.12,
                name=f"{_name} (full)", showlegend=False,
                hoverinfo="skip",
            ))

        if step > 0:
            fig.add_trace(go.Scatter3d(
                x=_coords[:step + 1, 0],
                y=_coords[:step + 1, 1],
                z=_coords[:step + 1, 2],
                mode="markers",
                marker=dict(
                    size=np.linspace(1.5, 4, step + 1),
                    color=np.arange(step + 1),
                    colorscale=[[0, _c0], [0.5, _c1], [1.0, _c2]],
                    opacity=0.75,
                    line=dict(width=0),
                ),
                name=_name,
                legendgroup=str(_class_id),
                hovertemplate=(
                    f"<b>{_name}</b><br>"
                    "Step: %{customdata}<br>"
                    "U1: %{x:.2f}<br>U2: %{y:.2f}<br>U3: %{z:.2f}"
                    "<extra></extra>"
                ),
                customdata=np.arange(step + 1),
            ))

        fig.add_trace(go.Scatter3d(
            x=[_coords[step, 0]], y=[_coords[step, 1]], z=[_coords[step, 2]],
            mode="markers",
            marker=dict(size=11, color=_c2, symbol="circle",
                        line=dict(color="white", width=1.5)),
            name=f"{_name} (now)",
            legendgroup=str(_class_id),
            showlegend=False,
            hovertemplate=f"<b>{_name}</b> — step {step}<extra></extra>",
        ))

        if show_final.value:
            fig.add_trace(go.Scatter3d(
                x=[_coords[-1, 0]], y=[_coords[-1, 1]], z=[_coords[-1, 2]],
                mode="markers+text",
                marker=dict(size=9, color=_c1, symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[_name.split()[0]],
                textposition="top center",
                textfont=dict(color="white", size=9),
                name=f"{_name} (final)",
                legendgroup=str(_class_id),
                showlegend=False,
                hovertemplate=f"<b>{_name}</b> — final image<extra></extra>",
            ))

        if show_vector.value and step < N - 1:
            _dx = _coords[step + 1, 0] - _coords[step, 0]
            _dy = _coords[step + 1, 1] - _coords[step, 1]
            _dz = _coords[step + 1, 2] - _coords[step, 2]
            fig.add_trace(go.Cone(
                x=[_coords[step, 0]], y=[_coords[step, 1]], z=[_coords[step, 2]],
                u=[_dx * 4.0], v=[_dy * 4.0], w=[_dz * 4.0],
                colorscale=[[0, _c1], [1, _c2]],
                showscale=False,
                sizemode="absolute", sizeref=0.4,
                name=f"{_name} (direction)",
                legendgroup=str(_class_id),
                showlegend=False,
                hovertemplate=f"<b>{_name}</b> — direction {step}→{step+1}<extra></extra>",
            ))

        if show_sphere.value:
            _r  = _sigma_cur * 0.5
            _u  = np.linspace(0, 2 * np.pi, 20)
            _v  = np.linspace(0, np.pi, 15)
            _cx, _cy, _cz = _coords[step]
            fig.add_trace(go.Surface(
                x=_cx + _r * np.outer(np.cos(_u), np.sin(_v)),
                y=_cy + _r * np.outer(np.sin(_u), np.sin(_v)),
                z=_cz + _r * np.outer(np.ones_like(_u), np.cos(_v)),
                opacity=0.06,
                colorscale=[[0, _c0], [1, _c1]],
                showscale=False,
                name=f"{_name} (σ={_sigma_cur:.3f})",
                legendgroup=str(_class_id),
                showlegend=False,
                hoverinfo="skip",
                contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgb(6,6,16)",
            xaxis=dict(showgrid=True, gridcolor="rgba(80,80,120,0.2)",
                       zeroline=False, showbackground=True,
                       backgroundcolor="rgba(10,10,25,0.6)", title="UMAP 1"),
            yaxis=dict(showgrid=True, gridcolor="rgba(80,80,120,0.2)",
                       zeroline=False, showbackground=True,
                       backgroundcolor="rgba(10,10,25,0.4)", title="UMAP 2"),
            zaxis=dict(showgrid=True, gridcolor="rgba(80,80,120,0.2)",
                       zeroline=False, showbackground=True,
                       backgroundcolor="rgba(10,10,25,0.2)", title="UMAP 3"),
            camera=dict(eye=dict(x=1.6, y=1.1, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        height=580,
        title=dict(
            text=f"Step {step} / {N-1}  —  Space: {METHOD}",
            font=dict(color="rgba(200,190,255,0.9)", size=13),
            x=0.02,
        ),
        legend=dict(
            font=dict(color="rgba(200,190,255,0.85)", size=11),
            bgcolor="rgba(10,10,30,0.7)",
            bordercolor="rgba(100,90,180,0.3)",
            borderwidth=0.5,
        ),
        uirevision="camera",
    )

    _left  = mo.vstack([controls, options_row], gap="1rem")
    layout = mo.hstack([_left, mo.ui.plotly(fig)], gap="2rem", widths=[1, 2])
    return (layout,)


@app.cell
def _(layout):
    layout
    return


@app.cell
def _(
    CLASSES,
    CLASS_COLORS,
    THUMB_EVERY,
    class_checkboxes,
    data,
    get_phase,
    mo,
    np,
    step_slider,
    thumbs_b64,
):
    _step      = step_slider.value
    _n_thumbs  = data[list(CLASSES.keys())[0]][1].shape[0]
    _thumb_idx = min(_step // THUMB_EVERY, _n_thumbs - 1)

    cards = []
    for _cid in CLASSES:
        if CLASSES[_cid] not in class_checkboxes.value:
            continue

        _coords, _, _sigmas = data[_cid]
        _c0, _c1, _     = CLASS_COLORS[_cid]
        _name           = CLASSES[_cid]
        _sigma          = float(_sigmas[_step])
        _noise_pct      = int(_sigma * 100)
        _phase_lbl, _pc = get_phase(_step, _cid)

        _dist_final = np.linalg.norm(_coords[_step] - _coords[-1])
        _dist_max   = max(np.linalg.norm(_coords[0] - _coords[-1]), 1e-8)
        _progress   = int((1 - _dist_final / _dist_max) * 100)

        _b64 = thumbs_b64[_cid][_thumb_idx]

        cards.append(mo.Html(f"""
        <div style="
            background:rgba(10,10,28,0.85);
            border:0.5px solid {_c0};
            border-radius:10px;
            padding:10px 12px;
            display:flex; align-items:center; gap:12px;
            min-width:230px;
        ">
            <img src="data:image/png;base64,{_b64}" width="72" height="72"
                 style="border-radius:6px; border:0.5px solid {_c0};
                        image-rendering:pixelated; flex-shrink:0;"
                 alt="{_name} step {_step}"/>
            <div style="flex:1; min-width:0;">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:5px;">
                    <span style="font-size:12px; font-weight:500; color:{_c1};
                                 white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{_name}</span>
                    <span style="font-size:10px; font-weight:500; color:{_pc};
                                 background:rgba(0,0,0,0.4); border:0.5px solid {_pc};
                                 border-radius:4px; padding:1px 6px; margin-left:6px;
                                 white-space:nowrap;">{_phase_lbl}</span>
                </div>
                <div style="font-size:11px; color:rgba(160,150,210,0.7);
                            margin-bottom:5px; font-family:monospace;">σ = {_sigma:.3f}</div>
                <div style="margin-bottom:4px;">
                    <div style="font-size:10px; color:rgba(140,130,190,0.5); margin-bottom:2px;">noise {_noise_pct}%</div>
                    <div style="background:rgba(30,20,50,0.8); border-radius:3px; height:3px; overflow:hidden;">
                        <div style="width:{_noise_pct}%; height:100%; background:{_c0}; border-radius:3px;"></div>
                    </div>
                </div>
                <div>
                    <div style="font-size:10px; color:rgba(140,130,190,0.5); margin-bottom:2px;">progress (UMAP) {_progress}%</div>
                    <div style="background:rgba(30,20,50,0.8); border-radius:3px; height:3px; overflow:hidden;">
                        <div style="width:{_progress}%; height:100%; background:{_c1}; border-radius:3px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """))

    mo.hstack(cards, gap="0.75rem", wrap=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 🎞️ Filmstrip — click on a frame to navigate
    """)
    return


@app.cell
def _(CLASSES, N, THUMB_EVERY, mo, set_step, step_slider, thumbs_b64):
    _ref_cid  = list(CLASSES.keys())[0]
    _b64_list = thumbs_b64[_ref_cid]
    _n_thumbs = len(_b64_list)

    _max_display = 15
    _stride      = max(1, _n_thumbs // _max_display)
    _disp_idxs   = list(range(0, _n_thumbs, _stride))

    # S'assurer que la toute dernière image est toujours incluse
    if _disp_idxs[-1] != _n_thumbs - 1:
        _disp_idxs.append(_n_thumbs - 1)

    _cur_thumb = min(step_slider.value // THUMB_EVERY, _n_thumbs - 1)

    # Fonction pour capturer correctement la valeur dans la boucle
    def make_handler(target):
        return lambda _: set_step(target)

    _cells = []
    for _i, _ti in enumerate(_disp_idxs):
        _step_for_thumb = min(_ti * THUMB_EVERY, N - 1)
    
        # Forcer la première image à 0% et la dernière à 100%
        if _i == 0:
            _pct = 0
        elif _i == len(_disp_idxs) - 1:
            _pct = 100
        else:
            _pct = int((_step_for_thumb / (N - 1)) * 100)
        
        _is_cur = abs(_ti - _cur_thumb) <= 1
        _border = "2px solid rgba(167,139,250,0.9)" if _is_cur else "2px solid transparent"
        _op     = "1.0" if _is_cur else "0.6"

        # Création du vrai bouton natif, mais avec un label vide !
        _btn = mo.ui.button(label=" ", on_change=make_handler(_step_for_thumb))

        # On place le bouton dans un conteneur invisible par-dessus l'image
        _cells.append(f"""
        <div class="filmstrip-item" style="display:inline-block; position:relative; margin:0 6px; transition: transform 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);"
             title="Étape {_step_for_thumb} ({_pct}%)">
        
            <!-- COUCHE INVISIBLE : Le bouton natif Marimo -->
            <div class="hidden-btn" style="position:absolute; top:0; left:0; width:100%; height:100%; z-index:10; opacity:0; overflow:hidden;">
                {_btn}
            </div>
        
            <!-- COUCHE VISUELLE : L'image et le texte -->
            <img src="data:image/png;base64,{_b64_list[_ti]}"
                 width="110" height="110"
                 style="border-radius:8px; border:{_border}; opacity:{_op};
                        display:block; image-rendering:pixelated; transition: all 0.2s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"/>
            <div style="font-size:14px; color:rgba(140,130,190,0.8); text-align:center; margin-top:6px; font-weight: 600; transition: color 0.2s ease;">{_pct}%</div>
        </div>
        """)

    _html_output = mo.Html(f"""
    <style>
    /* Scrollbar */
    .filmstrip-container::-webkit-scrollbar {{ height: 8px; }}
    .filmstrip-container::-webkit-scrollbar-track {{ background: rgba(30, 20, 50, 0.5); border-radius: 4px; }}
    .filmstrip-container::-webkit-scrollbar-thumb {{ background: rgba(100, 90, 180, 0.6); border-radius: 4px; }}
    .filmstrip-container::-webkit-scrollbar-thumb:hover {{ background: rgba(167, 139, 250, 0.8); }}

    /* Effets de Hover visuels */
    .filmstrip-item:hover {{ transform: scale(1.15) translateY(-4px); z-index: 10; }}
    .filmstrip-item:hover img {{ opacity: 1.0 !important; border-color: rgba(167, 139, 250, 0.9) !important; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6) !important; }}

    /* Ne pas colorer le bouton caché, seulement le texte visible */
    .filmstrip-item:hover div:not(.hidden-btn) {{ color: rgba(167, 139, 250, 1.0) !important; }}

    /* Forcer le bouton natif Marimo à prendre 100% de la zone de clic */
    .hidden-btn marimo-ui-element, .hidden-btn button {{
        width: 100% !important;
        height: 100% !important;
        min-height: 100% !important;
        min-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        cursor: pointer !important;
    }}
    </style>

    <div class="filmstrip-container" style="
        background:rgba(8,8,22,0.95);
        border:1px solid rgba(80,70,140,0.4);
        border-radius:12px;
        padding:20px 16px 12px;
        overflow-x:auto;
        white-space:nowrap;
        display:flex;
        align-items:center;
    ">{''.join(_cells)}</div>
    """)

    _html_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 📈 Velocity & Convergence in Embedded Space
    """)
    return


@app.cell
def _(CLASSES, CLASS_COLORS, N, data, go, mo, np, speeds_all, step_slider):
    _step_v = step_slider.value

    # ── Velocity Chart ────────────────────────────────────────────────────
    fig_v = go.Figure()
    for _cid, _cname in CLASSES.items():
        _, _c1, _ = CLASS_COLORS[_cid]
        _sp = speeds_all[_cid]
        fig_v.add_trace(go.Scatter(
            x=list(range(len(_sp))), y=_sp.tolist(),
            mode="lines", name=_cname,
            line=dict(color=_c1, width=1.8), opacity=0.9,
        ))
    fig_v.add_vline(
        x=_step_v,
        line=dict(color="rgba(255,255,255,0.5)", dash="dot", width=1.2),
        annotation_text=f"step {_step_v}",
        annotation_font_color="rgba(200,190,255,0.8)",
        annotation_font_size=10,
    )
    fig_v.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,6,16,0.95)",
        height=190, margin=dict(l=44, r=10, t=28, b=32),
        title=dict(text="Velocity ‖Δx_t‖ in UMAP space",
                   font=dict(color="rgba(200,190,255,0.8)", size=12), x=0.01),
        xaxis=dict(title="Step", color="rgba(160,150,210,0.6)",
                   gridcolor="rgba(80,80,120,0.15)", range=[0, N-2]),
        yaxis=dict(title="‖Δx‖ UMAP", color="rgba(160,150,210,0.6)",
                   gridcolor="rgba(80,80,120,0.15)"),
        legend=dict(font=dict(color="rgba(200,190,255,0.75)", size=10),
                    bgcolor="rgba(10,10,30,0.6)", x=1.0, xanchor="right"),
        clickmode="event",
    )
    velocity_chart = mo.ui.plotly(fig_v)

    # ── Distance to Final Chart ──────────────────────────────────────────
    fig_d = go.Figure()
    for _cid, _cname in CLASSES.items():
        _c0, _c1, _ = CLASS_COLORS[_cid]
        _coords, _, _ = data[_cid]
        _dists = np.linalg.norm(_coords - _coords[-1], axis=1)
        fig_d.add_trace(go.Scatter(
            x=list(range(len(_dists))), y=_dists.tolist(),
            mode="lines", name=_cname,
            line=dict(color=_c1, width=1.8),
            fill="tozeroy", fillcolor=_c0,
            opacity=0.9,
        ))
    fig_d.add_vline(
        x=_step_v,
        line=dict(color="rgba(255,255,255,0.5)", dash="dot", width=1.2),
    )
    fig_d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,6,16,0.95)",
        height=190, margin=dict(l=44, r=10, t=28, b=32),
        title=dict(text="Distance to final point ‖x_t − x_T‖ in UMAP space",
                   font=dict(color="rgba(200,190,255,0.8)", size=12), x=0.01),
        xaxis=dict(title="Step", color="rgba(160,150,210,0.6)",
                   gridcolor="rgba(80,80,120,0.15)", range=[0, N-1]),
        yaxis=dict(title="‖x_t − x_T‖ UMAP", color="rgba(160,150,210,0.6)",
                   gridcolor="rgba(80,80,120,0.15)"),
        legend=dict(font=dict(color="rgba(200,190,255,0.75)", size=10),
                    bgcolor="rgba(10,10,30,0.6)", x=1.0, xanchor="right"),
    )

    display_charts = mo.vstack([
        mo.hstack([velocity_chart, mo.ui.plotly(fig_d)], gap="1rem"),
    ])
    return display_charts, velocity_chart


@app.cell
def _(display_charts):
    display_charts
    return


@app.cell
def _(
    CLASSES,
    CLASS_COLORS,
    THUMB_EVERY,
    data,
    mo,
    thumbs_b64,
    velocity_chart,
):
    _click = velocity_chart.value

    if _click and _click.get("points"):
        _clicked_step = int(_click["points"][0].get("x", 0))
        _n_thumbs     = data[list(CLASSES.keys())[0]][1].shape[0]
        _tidx         = min(_clicked_step // THUMB_EVERY, _n_thumbs - 1)

        _panels = []
        for _cid, _cname in CLASSES.items():
            _c0, _c1, _ = CLASS_COLORS[_cid]
            _, _, _sigmas = data[_cid]
            _sigma = float(_sigmas[min(_clicked_step, len(_sigmas) - 1)])
            _b64   = thumbs_b64[_cid][_tidx]
            _panels.append(mo.Html(f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{_b64}" width="88" height="88"
                     style="border-radius:6px; border:1px solid {_c0};
                            image-rendering:pixelated; display:block; margin:0 auto 4px"/>
                <div style="font-size:10px; color:{_c1}; white-space:nowrap;">{_cname.split()[0]}</div>
                <div style="font-size:9px; color:rgba(140,130,190,0.6); font-family:monospace;">σ={_sigma:.3f}</div>
            </div>
            """))

        mo.vstack([
            mo.md(f"**Images at step {_clicked_step}**"),
            mo.hstack(_panels, gap="1.2rem", wrap=True),
        ])
    else:
        mo.Html(
            """
        <div style="color:rgba(140,130,190,0.35); font-size:12px;
                    font-style:italic; padding:6px 0;">
            Click on the velocity chart to reveal the intermediate images for that specific step.
        </div>
        """
        )
    return


@app.cell
def _(CLASSES, METHOD, N, data, get_phase, mo, np, speeds_all, step_slider):
    _step_m = step_slider.value

    _table_data = []
    for _cid, _cname in CLASSES.items():
        _coords, _, _sigmas = data[_cid]
        _sigma  = float(_sigmas[_step_m])
        _speed  = float(speeds_all[_cid][min(_step_m, len(speeds_all[_cid]) - 1)])
        _dist   = float(np.linalg.norm(_coords[_step_m] - _coords[-1]))
        _phase, _ = get_phase(_step_m, _cid)

        _table_data.append({
            "Class": _cname,
            "Current σ": round(_sigma, 4),
            "Velocity ‖Δx‖ (UMAP)": round(_speed, 4),
            "Distance to final (UMAP)": round(_dist, 4),
            "Phase": _phase
        })

    _layout = mo.vstack([
        mo.md(f"### Metrics — step {_step_m} / {N-1}"),
        mo.ui.table(_table_data),
        mo.md(f"> **Projection space:** {METHOD}")
    ])

    _layout
    return


if __name__ == "__main__":
    app.run()
