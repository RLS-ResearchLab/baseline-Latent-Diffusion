"""
ViT-based VAE + DiT (diffusion transformer), both using
2D axial RoPE for position embedding

+ factory a function
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# RoPE (pos embed) utility functions: read apply_rope first

def build_2d_rope_cache(h, w, head_dim, base=10000.0, device=None):
    """
    # in 1D rope (pos embedding a vector position in a 1D sequence)
    # we rotate each pair of values in the vec following a given angle (=> should be pair dim vec)
    # for images, tokens are in a 2D grid
    # we embed position along the width axis (x) in the first half of the vector
    # we embed position along the height axis (y) in the second half of the vector
    # => vec dim should be divisible by 4
    # pos embedding is applied on each head vec (a $head_dim$ values chunk of the token/vec) 
    """
    
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    freq_dim = head_dim // 4
    inv_freq = 1.0 / (base ** (torch.arange(0, freq_dim, device=device).float() / freq_dim))
    # 1, .998, .985, ...

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device).float(),
        torch.arange(w, device=device).float(),
        indexing="ij",
    )
    
    ys, xs = ys.reshape(-1), xs.reshape(-1) 
    # x position of each token in the grid: 0, 1, 2, ..., H'-1, 0, 1, ..., H'-1, ...
    # y position of each token in the grid: 0, 0, ..., 1, 1, ..., 2, 2, ..., ...

    freqs_x = torch.outer(xs, inv_freq)
    freqs_y = torch.outer(ys, inv_freq)
    # (y_0 * inv_freq_0), (y_0 * inv_freq_1), (y_0 * inv_freq_2), ...
    # (y_1 * inv_freq_0), (y_1 * inv_freq_1), (y_1 * inv_freq_2), ...
    # ...

    # first half of the vec rotated following x_pos, second half rotated following y_pos
    freqs = torch.cat([freqs_x, freqs_y], dim=-1) 
    
    # since we "rotate pairs of values", 
    # meaning each pair (not adjascent pairs) of values in the vec are rotate using the same angle, 
    # we assign the same freq / angle to both values
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()



def rotate_half(x):
    """
    remember how we precomputed / stored rotation vectors in build_2d_rope_cache: [freqs, freqs]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """    
    the final goal of rope is to apply this matrix (rotation) to each pair of values (x[i] and x[i+dim/2])
    [ out[i]       ]   [ cos θ   -sin θ ] [ x[i]       ]
    [ out[i+dim/2] ] = [ sin θ    cos θ ] [ x[i+dim/2] ]


    in implementation it's decomposed to : 
    out[i]         = x[i]        * cos(θ) + (-x[i+dim/2]) * sin(θ)
    out[i+dim/2]   = x[i+dim/2]  * cos(θ) + ( x[i])        * sin(θ)
    """

    # x: (B, heads, N, head_dim), cos/sin: (N, head_dim)
    # cos/sin: (N, head_dim), cos/sin[None, None]: (_, _, N, head_dim)

    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class RoPEAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim) # atten heads output aggregation

    def forward(self, x, rope_cos, rope_sin):
        B, N, C = x.shape  # (batch, tokens, dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        # position embedding using rope
        # query and key only, again, disentangling representation and information: v and x, from position and communication: q and k
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # attention matrix, softmax => attention scores, weight V, scale by sqrt(d); per head
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(
            self.act(
                self.fc1(x)
            )
        )


def patchify(x, patch_size):
    """
    image (pixel grid) => token sequence
    C, H, W => N, patch_dim; per sample in in batch
    """
    B, C, H, W = x.shape
    p = patch_size
    x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, H/p, W/p, C, p, p)
    return x.view(B, (H // p) * (W // p), C * p * p)  # (B, N, patch_dim)


def unpatchify(x, patch_size, img_size, channels):
    """
    token sequence  => image (pixel grid)
    N, patch_dim => C, H, W; per sample in in batch
    """
    B, N, _ = x.shape
    p = patch_size
    hw = img_size // p
    x = x.view(B, hw, hw, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, H/p, p, W/p, p)
    return x.view(B, channels, img_size, img_size)


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ViTBlock(dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, rope_cos, rope_sin):
        """
        to eliminate redundancy, sinse the number of tokens is fixed (mostly)
        we precompute the rope angles once and cache / save them to reuse instead of recomputing them each pass
        """

        for blk in self.blocks:
            x = blk(x, rope_cos, rope_sin)
        return self.norm(x)


class VAE_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, latent_dim, embed_dim, depth, heads):
        super().__init__()
        self.patch_size = patch_size
        grid = img_size // patch_size
        patch_dim = in_ch * patch_size * patch_size

        self.proj_in = nn.Linear(patch_dim, embed_dim)
        cos, sin = build_2d_rope_cache(grid, grid, embed_dim // heads)

        # model tensors are categorized into: buffers and params
        # params are optimizable (grad computation)
        # buffers are fixed and cached, saved in model checkpoint unless we set persistent=False
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.backbone = ViT(embed_dim, depth, heads)
        self.to_mu = nn.Linear(embed_dim, latent_dim)
        self.to_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        x = self.proj_in(patchify(x, self.patch_size))  # (B, N, embed_dim)
        x = self.backbone(x, self.rope_cos, self.rope_sin)
        return self.to_mu(x), self.to_logvar(x)  # each (B, N, latent_dim)


class VAE_Decoder(nn.Module):
    def __init__(self, img_size, patch_size, out_ch, latent_dim, embed_dim, depth, heads):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_ch = out_ch
        grid = img_size // patch_size
        patch_dim = out_ch * patch_size * patch_size

        self.proj_in = nn.Linear(latent_dim, embed_dim)
        cos, sin = build_2d_rope_cache(grid, grid, embed_dim // heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.backbone = ViT(embed_dim, depth, heads)
        self.proj_out = nn.Linear(embed_dim, patch_dim)

    def forward(self, z):
        x = self.proj_in(z)  # (B, N, embed_dim)
        x = self.backbone(x, self.rope_cos, self.rope_sin)
        x = self.proj_out(x)  # (B, N, patch_dim)
        return unpatchify(x, self.patch_size, self.img_size, self.out_ch)


class VAE(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_ch=3, latent_dim=32,
                 embed_dim=384, enc_depth=8, dec_depth=8, heads=6):
        super().__init__()
        self.encoder = VAE_Encoder(img_size, patch_size, in_ch, latent_dim, embed_dim, enc_depth, heads)
        self.decoder = VAE_Decoder(img_size, patch_size, in_ch, latent_dim, embed_dim, dec_depth, heads)

    @staticmethod
    def reparameterize(mu, logvar):
        """
        to keep the sampled latent differentiable, we use this trick
        instead of sampling eps ~ N(mu, std)
        we sample N(0, I) and use std * eps + mu
        which deem the output diffirentiable w.r.t std and mu 
        """
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x, sample=True):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar) if sample else mu

    def decode(self, z):
        return self.decoder(z)


# adaLN-Zero conditioned ViT blocks: timestep + class embedding modulate


class TimestepEmbedder(nn.Module):
    """
    timestep_vec = sinusoidal(timestep_value)
    timestep_embedding = MLP(timestep_vec)

    don't bother much with the sinusoidal logic


    *question: why don't we precompute and cache / store these like we did with rope
    """
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    @staticmethod
    def sinusoidal(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):
        return self.mlp(self.sinusoidal(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, dim)  # +1 = null class for CFG
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train=False):
        if train and self.dropout_prob > 0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob

            labels = torch.where(drop, self.num_classes, labels)
            # use class_id (0 -> N-1) embedding for a given label              : sample from class i
            # use no_class (N) embedding when we're not passing it to the model: sample could be any class

        return self.embedding(labels)


def modulate(x, shift, scale):
    """
    this is the normalizaiton scale and shift utility func
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()

        # nn.LayerNorm by default first normalize (using inputs mean and std) to 0 mean and unit var
        # if elementwise_affine = True
        # then scale and shit using learnable values (x_norm * learned_scale + learned_shift)
        # since we're using adaIN style conditioning (i.e the scale and shift depend on the conditions)
        # we use nn.LayerNorm only to normalize to 0 mean and unit var and "modulate" (or scale and shift)
        # using the computed conditioned_scale and conditioned_shift vectors, we then gate
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = RoPEAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio)

        # scale, shift, gate (3) of the both normlayer s (6 total)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        # initially using scale = shift = gate = 0
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)  # zero-init -> block starts as identity

    def forward(self, x, cond, rope_cos, rope_sin):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(cond).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift_a, scale_a)
        x = x + gate_a.unsqueeze(1) * self.attn(h, rope_cos, rope_sin)

        h = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.proj = nn.Linear(dim, out_dim, bias=True)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, grid_size=16, latent_dim=32, dim=768, depth=12, heads=12,
                 mlp_ratio=4.0, num_classes=1000, class_dropout_prob=0.1):
        super().__init__()
        self.token_in = nn.Linear(latent_dim, dim)
        self.t_embed = TimestepEmbedder(dim)
        self.y_embed = LabelEmbedder(num_classes, dim, class_dropout_prob)

        cos, sin = build_2d_rope_cache(grid_size, grid_size, dim // heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(dim, latent_dim)  # predict noise/velocity in latent_dim

    def forward(self, z, t, y, train=True):
        # z: (B, N, latent_dim) noisy latent tokens, t: (B,), y: (B,) class ids
        x = self.token_in(z)  # (B, N, dim)
        cond = self.t_embed(t) + self.y_embed(y, train=train)  # (B, dim)
        for blk in self.blocks:
            x = blk(x, cond, self.rope_cos, self.rope_sin)
        return self.final(x, cond)  # (B, N, latent_dim)



def build_model(model_type: str, config: dict = None):
    """Factory: model_type in {'vae', 'dit'} -> ViTVAE or DiT instance."""
    config = config or {}
    if model_type == "vae":
        return VAE(**config)
    if model_type == "dit":
        return DiT(**config)
    raise ValueError(f"unknown model_type: {model_type!r}, expected 'vae' or 'dit'")
