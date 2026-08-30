# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor, window_origin: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition. Capture B explicitly so window_unpartition doesn't
        # have to recover it from a symbolic floor-divide — torch.export's
        # shape solver records that divide as a constant-batch guard
        # (see commit history for details).
        if self.window_size > 0:
            origin = window_origin if window_origin is not None else (0, 0)
            B, H, W = x.shape[0], x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size, origin)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W), B, origin)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        rest = dict()
        for k, v in state_dict.items():
            if 'rel_pos' in k:
                my_rel_pos = getattr(self, k[len(prefix):])
                if my_rel_pos.shape[0] != v.shape[0]:
                    v = v.unsqueeze(0).permute(0, 2, 1)
                    v = F.interpolate(v, size=my_rel_pos.shape[0], mode='linear', align_corners=True)
                    v = v.squeeze(0).T
                my_rel_pos.data.copy_(v)
            else:
                rest[k] = v

        return super()._load_from_state_dict(rest, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)


def window_partition(
    x: torch.Tensor, window_size: int, origin: Tuple[int, int] = (0, 0)
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
        origin (tuple): (oy, ox) share of each axis's pad budget allocated to the
            top/left instead of the bottom/right. Shifts the window-partition
            phase relative to content WITHOUT adding pad: total padding stays
            (window_size - dim % window_size) % window_size per axis, so the
            window count is unchanged. Must satisfy 0 <= oy <= pad_h (resp. ox).

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    oy, ox = origin
    assert 0 <= oy <= pad_h and 0 <= ox <= pad_w, \
        f"window origin {origin} exceeds pad budget ({pad_h}, {pad_w})"
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, ox, pad_w - ox, oy, pad_h - oy))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
    B: int,
    origin: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
        B (int): original batch size, passed in by the caller. Originally recovered
            via `windows.shape[0] // (Hp * Wp // ws // ws)`, but torch.export's
            shape solver records the floor-divide as a constant-batch guard,
            blocking dynamic-batch engines. Passing B keeps the dim symbolic.
        origin (tuple): the (oy, ox) pad split used by the matching
            ``window_partition`` call.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    oy, ox = origin
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W or oy > 0 or ox > 0:
        x = x[:, oy:oy + H, ox:ox + W, :].contiguous()
    return x


def get_abs_pos(pos_embed: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Absolute position embedding for an arbitrary token-grid size.

    Grids no larger than the table use its top-left crop (ViTDet convention);
    larger grids bilinearly interpolate it.

    Args:
        pos_embed (Tensor): position embedding table with [1, Hp, Wp, C].
        hw (Tuple): target token grid (h, w).

    Returns:
        Position embedding with [1, h, w, C].
    """
    h, w = hw
    if (h, w) == tuple(pos_embed.shape[1:3]):
        return pos_embed
    if h <= pos_embed.shape[1] and w <= pos_embed.shape[2]:
        return pos_embed[:, :h, :w]
    pe = pos_embed.permute(0, 3, 1, 2)
    pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
    return pe.permute(0, 2, 3, 1)


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    # Built on rel_pos's device: CPU coords would make the gather below index
    # a CUDA tensor with a CPU index — an implicit synchronizing H2D copy on
    # every attention block.
    q_coords = torch.arange(q_size, device=rel_pos_resized.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos_resized.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def _packed_block_forward(
    blk: Block,
    x_flat: torch.Tensor,
    sizes: List[Tuple[int, int]],
    offsets: List[int],
    origins: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """One Block forward over a packed sequence of per-image token grids.

    Same math as running each image's grid through ``blk`` alone. Token-wise ops
    (norms, MLP, residuals) run on the packed [total, C] sequence in one call.
    Windowed attention batches every image's (identical-size, zero-padded)
    windows into a single ``blk.attn`` call — all windows share the same
    window-size relative-position tables. Global attention runs one call per
    distinct grid size, because the decomposed relative-position bias is
    interpolated per (h, w).

    ``origins``: optional per-image (oy, ox) pad splits for the window
    partition (see ``window_partition``); the same list must be used for every
    block of a forward so all blocks share one partition phase per image.
    """
    shortcut = x_flat
    x = blk.norm1(x_flat)

    if blk.window_size > 0:
        win_batches: List[torch.Tensor] = []
        metas: List[Tuple[int, Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
        for i, ((h, w), offset) in enumerate(zip(sizes, offsets)):
            origin = origins[i] if origins is not None else (0, 0)
            xi = x[offset : offset + h * w].view(1, h, w, -1)
            wins, pad_hw = window_partition(xi, blk.window_size, origin)
            win_batches.append(wins)
            metas.append((wins.shape[0], pad_hw, (h, w), origin))
        wins_all = blk.attn(torch.cat(win_batches, dim=0))
        parts: List[torch.Tensor] = []
        woffset = 0
        for nwin, pad_hw, hw, origin in metas:
            xi = window_unpartition(wins_all[woffset : woffset + nwin], blk.window_size, pad_hw, hw, 1, origin)
            parts.append(xi.reshape(-1, xi.shape[-1]))
            woffset += nwin
        x = torch.cat(parts, dim=0)
    else:
        groups: Dict[Tuple[int, int], List[int]] = {}
        for i, hw in enumerate(sizes):
            groups.setdefault(hw, []).append(i)
        parts = [None] * len(sizes)  # type: ignore[list-item]
        for (h, w), idxs in groups.items():
            xi = torch.stack([x[offsets[i] : offsets[i] + h * w].view(h, w, -1) for i in idxs], dim=0)
            xi = blk.attn(xi)
            for j, i in enumerate(idxs):
                parts[i] = xi[j].reshape(h * w, -1)
        x = torch.cat(parts, dim=0)

    x = shortcut + x
    return x + blk.mlp(blk.norm2(x))


def sample_window_origins(
    sizes: List[Tuple[int, int]], window_size: int
) -> List[Tuple[int, int]]:
    """Sample a per-image window-partition phase within the existing pad budget.

    For each axis the pad budget is (ws - dim % ws) % ws; the top/left share is
    drawn uniformly from [0, budget], so total padding (and window count) is
    unchanged — only the partition phase relative to content moves. An axis
    whose grid is an exact multiple of ws has zero budget and stays at phase 0.
    """
    origins: List[Tuple[int, int]] = []
    for h, w in sizes:
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        oy = int(torch.randint(0, pad_h + 1, ())) if pad_h else 0
        ox = int(torch.randint(0, pad_w + 1, ())) if pad_w else 0
        origins.append((oy, ox))
    return origins


def forward_trunk_packed(
    patch_embed: nn.Module,
    pos_embed: Optional[torch.Tensor],
    blocks: nn.ModuleList,
    images: List[torch.Tensor],
    window_phase_jitter: bool = False,
) -> List[torch.Tensor]:
    """Run the ViTDet trunk (patch_embed -> abs pos -> blocks, no neck) over a
    list of variable-size images packed into one token sequence.

    Function-preserving versus forwarding each image alone: every image gets its
    own absolute-position crop/interpolation (``get_abs_pos``) and its own
    relative-position bias in global-attention blocks, and windows never span
    images. See ``_packed_block_forward`` for how attention is batched.

    Args:
        patch_embed (nn.Module): conv patch embedding, [B, 3, H, W] -> [B, h, w, C].
        pos_embed (Tensor or None): absolute position table with [1, Hp, Wp, C].
        blocks (nn.ModuleList): the trunk's ``Block`` list.
        images (list): image tensors with [3, H_i, W_i] or [1, 3, H_i, W_i].

    Returns:
        One [h_i, w_i, C] feature map per image (pre-neck).
    """
    sizes: List[Tuple[int, int]] = []
    flat_tokens: List[torch.Tensor] = []
    for img in images:
        if img.ndim == 3:
            img = img.unsqueeze(0)
        x = patch_embed(img)  # [1, h, w, C]
        if pos_embed is not None:
            x = x + get_abs_pos(pos_embed, (x.shape[1], x.shape[2]))
        sizes.append((x.shape[1], x.shape[2]))
        flat_tokens.append(x.reshape(-1, x.shape[-1]))

    offsets = [0]
    for h, w in sizes:
        offsets.append(offsets[-1] + h * w)

    # Window-phase jitter: one (oy, ox) pad split per image, shared by every
    # windowed block of this forward, so the ViTDet window grid's fixed-pattern
    # boundary artifact lands at a random phase relative to content (within the
    # existing pad budget -- no extra windows).
    origins: Optional[List[Tuple[int, int]]] = None
    if window_phase_jitter:
        win_sizes = {blk.window_size for blk in blocks if blk.window_size > 0}
        if len(win_sizes) > 1:
            raise ValueError(f"window_phase_jitter requires a single window size, got {sorted(win_sizes)}")
        if win_sizes:
            origins = sample_window_origins(sizes, next(iter(win_sizes)))

    x_flat = torch.cat(flat_tokens, dim=0)  # [total, C]
    for blk in blocks:
        x_flat = _packed_block_forward(blk, x_flat, sizes, offsets, origins)

    return [
        x_flat[offset : offset + h * w].view(h, w, -1)
        for (h, w), offset in zip(sizes, offsets)
    ]
