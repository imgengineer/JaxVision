from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ..ops.misc import MLP
from ..ops.stochastic_depth import StochasticDepth

__all__ = [
    "SwinTransformer",
    "swin_b",
    "swin_s",
    "swin_t",
    "swin_v2_b",
    "swin_v2_s",
    "swin_v2_t",
]


def _patch_merging_pad(x: jax.Array) -> jax.Array:
    h, w, _ = x.shape[1:]
    x = jnp.pad(
        x,
        (
            (0, 0),
            (0, h % 2),
            (0, w % 2),
            (0, 0),
        ),
    )
    x0 = x[..., 0::2, 0::2, :]
    x1 = x[..., 1::2, 0::2, :]
    x2 = x[..., 0::2, 1::2, :]
    x3 = x[..., 1::2, 1::2, :]
    return jnp.concat([x0, x1, x2, x3], axis=-1)


def _get_relative_position_bias(
    relative_position_bias_table: jax.Array,
    relative_position_index: jax.Array,
    window_size: list[int],
) -> jax.Array:
    n = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.reshape(n, n, -1)
    return jnp.expand_dims(relative_position_bias.transpose(2, 0, 1), axis=0)


class PatchMerging(nnx.Module):
    """Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.

    """

    def __init__(
        self,
        dim: int,
        norm_layer: Callable[..., nnx.Module] = nnx.LayerNorm,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False, rngs=rngs)
        self.norm = norm_layer(4 * dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Args:
            x (jax.Array): input tensor with expected layout of [..., H, W, C].

        Returns:
            jax.Array with layout of [..., H/2, W/2, 2*C]

        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        return self.reduction(x)


class PatchMergingV2(nnx.Module):
    """Patch Merging Layer for Swin Transformer V2.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.

    """

    def __init__(self, dim: int, norm_layer: Callable[..., nnx.Module] = nnx.LayerNorm, *, rngs: nnx.Rngs):
        self.dim = dim
        self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False, rngs=rngs)
        self.norm = norm_layer(2 * dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Args:
            x (jax.Array): input tensor with expected layout of [..., H, W, C].

        Returns:
            jax.Array with layout of [..., H/2, W/2, 2*C]

        """
        x = _patch_merging_pad(x)
        x = self.reduction(x)
        return self.norm(x)


def shifted_window_attention(
    inputs: jax.Array,
    qkv_weight: jax.Array,
    proj_weight: jax.Array,
    relative_position_bias: jax.Array,
    window_size: list[int],
    num_heads: int,
    shift_size: list[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: jax.Array | None = None,
    proj_bias: jax.Array | None = None,
    logit_scale: jax.Array | None = None,
    *,
    deterministic: bool = False,
    rngs: nnx.Rngs,
) -> jax.Array:
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        input (jax.Array[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (jax.Array[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (jax.Array[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (jax.Array): The learned relative position bias added to attention.
        window_size (list[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (list[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.

    Returns:
        jax.Array[N, H, W, C]: The output tensor after shifted window attention.

    """
    b, h_orig, w_orig, c = inputs.shape

    pad_r = (window_size[1] - w_orig % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - h_orig % window_size[0]) % window_size[0]
    x = jnp.pad(
        inputs,
        (
            (0, 0),
            (0, pad_r),
            (0, pad_b),
            (0, 0),
        ),
    )
    _, pad_h, pad_w, _ = x.shape

    shift_size = shift_size.copy()

    if window_size[0] >= pad_h:
        shift_size[0] = 0
    if window_size[1] >= pad_w:
        shift_size[1] = 0

    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(-shift_size[0], -shift_size[1]), axis=(1, 2))

    num_windows = (pad_h // window_size[0]) * (pad_w // window_size[1])
    x = x.reshape(b, pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1], c)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(b * num_windows, window_size[0] * window_size[1], c)

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = jnp.copy(qkv_bias)
        length = qkv_bias.size // 3
        qkv_bias = qkv_bias.at[length : 2 * length].set(0.0)
    qkv = x @ qkv_weight + qkv_bias
    qkv = qkv.reshape(x.shape[0], x.shape[1], 3, num_heads, c // num_heads).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        attn = jnp.linalg.norm(q, axis=-1, keepdims=True) @ jnp.linalg.norm(k, axis=-1, keepdims=True).transpose(0, 1, 3, 2)
        logit_scale = jnp.exp(jnp.clip(logit_scale, max=jnp.log(100.0)))
        attn = attn * logit_scale
    else:
        q = q * (c // num_heads) ** -0.5
        attn = q @ (k.transpose(0, 1, 3, 2))

    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = jnp.zeros((pad_h, pad_w))
        h_slices_for_mask = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices_for_mask = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for hs in h_slices_for_mask:
            for ws in w_slices_for_mask:
                attn_mask = attn_mask.at[hs[0] : hs[1], ws[0] : ws[1]].set(count)
                count += 1
        attn_mask = attn_mask.reshape(pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1])
        attn_mask = attn_mask.transpose(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = jnp.expand_dims(attn_mask, axis=1) - jnp.expand_dims(attn_mask, axis=2)
        attn_mask = jnp.where(attn_mask != 0, (-100.0), 0.0)
        attn = attn.reshape(x.shape[0] // num_windows, num_windows, num_heads, x.shape[1], x.shape[1])
        attn = attn + jnp.expand_dims(jnp.expand_dims(attn_mask, axis=1), axis=0)
        attn = attn.reshape(-1, num_heads, x.shape[1], x.shape[1])

    attn = nnx.softmax(attn, axis=-1)
    attn = nnx.Dropout(rate=attention_dropout, rngs=rngs, deterministic=deterministic)(attn)

    x = (attn @ v).transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], c)
    x = x @ proj_weight + proj_bias
    x = nnx.Dropout(rate=dropout, rngs=rngs, deterministic=deterministic)(x)

    x = x.reshape(b, pad_h // window_size[0], pad_w // window_size[1], window_size[0], window_size[1], c)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(b, pad_h, pad_w, c)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(shift_size[0], shift_size[1]), axis=(1, 2))

    return x[:, :h_orig, :w_orig, :]


class ShiftedWindowAttention(nnx.Module):
    """See :func:`shifted_window_attention`."""

    def __init__(
        self,
        dim: int,
        window_size: list[int],
        shift_size: list[int],
        num_heads: int,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        *,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        deterministic: bool = False,
        rngs: nnx.Rngs,
    ):
        if len(window_size) != 2 or len(shift_size) != 2:
            msg = "window_size and shift_size must be of length 2"
            raise ValueError(msg)
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.deterministic = deterministic
        self.rngs = rngs

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, use_bias=proj_bias, rngs=rngs)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nnx.Param(
            jnp.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)),
        )
        self.relative_position_bias_table.values = nnx.initializers.truncated_normal(stddev=0.02)(
            self.rngs.params(),
            shape=self.relative_position_bias_table.value.shape,
        )

    def define_relative_position_index(self):
        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(1, 2, 0)
        relative_coords = relative_coords.at[:, :, 0].add(self.window_size[0] - 1)
        relative_coords = relative_coords.at[:, :, 1].add(self.window_size[1] - 1)
        relative_coords = relative_coords.at[:, :, 0].mul(2 * self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(axis=-1).flatten()
        self.relative_position_index = relative_position_index

    def get_relative_position_bias(self) -> jax.Array:
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,
            self.window_size,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Args:
            x (Tensor): Tensor with layout of [B, H, W, C].

        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]

        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.kernel.value,
            self.proj.kernel.value,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias.value,
            proj_bias=self.proj.bias.value,
            deterministic=self.deterministic,
            rngs=self.rngs,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """See :func:`shifted_window_attention_v2`."""

    def __init__(
        self,
        dim: int,
        window_size: list[int],
        shift_size: list[int],
        num_heads: int,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        *,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        deterministic: bool = False,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
            deterministic=deterministic,
            rngs=rngs,
        )

        self.logit_scale = nnx.Param(jnp.log(10 * jnp.ones((num_heads, 1, 1))))

        self.cpb_mlp = nnx.Sequential(
            nnx.Linear(2, 512, use_bias=True, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, num_heads, use_bias=False, rngs=rngs),
        )
        if qkv_bias:
            length = self.qkv.bias.value.size // 3
            self.qkv.bias.value = self.qkv.bias.value.at[length : 2 * length].set(0.0)

    def define_relative_position_bias_table(self):
        relative_coords_h = jnp.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=jnp.float32)
        relative_coords_w = jnp.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=jnp.float32)
        relative_coords_table = jnp.stack(jnp.meshgrid(relative_coords_h, relative_coords_w, indexing="ij"))
        relative_coords_table = jnp.expand_dims(
            relative_coords_table.transpose(1, 2, 0),
            axis=0,
        )

        relative_coords_table = relative_coords_table.at[:, :, :, 0].set(
            relative_coords_table[:, :, :, 0] / (self.window_size[0] - 1),
        )
        relative_coords_table = relative_coords_table.at[:, :, :, 1].set(
            relative_coords_table[:, :, :, 1] / (self.window_size[1] - 1),
        )

        relative_coords_table *= 8
        relative_coords_table = jnp.sign(relative_coords_table) * jnp.log2(jnp.abs(relative_coords_table) + 1.0) / 3.0
        self.relative_coords_table = relative_coords_table

    def get_relative_position_bias(self) -> jax.Array:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).reshape(-1, self.num_heads),
            self.relative_position_index,
            self.window_size,
        )
        return 16 * nnx.sigmoid(relative_position_bias)

    def __call__(self, x: jax.Array):
        """Args:
            x (Tensor): Tensor with layout of [B, H, W, C].

        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]

        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.kernel.value,
            self.proj.kernel.value,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias.value,
            proj_bias=self.proj.bias.value,
            logit_scale=self.logit_scale.value,
            deterministic=self.deterministic,
            rngs=self.rngs,
        )


class SwinTransformerBlock(nnx.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (list[int]): Window size.
        shift_size (list[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention

    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        shift_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nnx.Module] = nnx.LayerNorm,
        attn_layer: Callable[..., nnx.Module] = ShiftedWindowAttention,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = norm_layer(dim, rngs=rngs)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
            rngs=rngs,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row", rngs=rngs)
        self.norm2 = norm_layer(dim, rngs=rngs)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nnx.gelu, dropout=dropout, rngs=rngs)

        for _, m in self.mlp.iter_modules():
            if isinstance(m, nnx.Linear):
                m.kernel_init = nnx.initializers.xavier_uniform()
                if m.bias is not None:
                    m.bias_init = nnx.initializers.normal(stddev=1e-6)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        return x + self.stochastic_depth(self.mlp(self.norm2(x)))


class SwinTransformerBlockV2(SwinTransformerBlock):
    """Swin Transformer V2 Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (list[int]): Window size.
        shift_size (list[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttentionV2.

    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        shift_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nnx.Module] = nnx.LayerNorm,
        attn_layer: Callable[..., nnx.Module] = ShiftedWindowAttentionV2,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        return x + self.stochastic_depth(self.norm2(self.mlp(x)))


class SwinTransformer(nnx.Module):
    """Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.

    Args:
        patch_size (list[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (list(int)): Depth of each Swin Transformer layer.
        num_heads (list(int)): Number of attention heads in different layers.
        window_size (list[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.

    """

    def __init__(
        self,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        num_heads: list[int],
        window_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Callable[..., nnx.Module] | None = None,
        block: Callable[..., nnx.Module] | None = None,
        downsample_layer: Callable[..., nnx.Module] = PatchMerging,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nnx.LayerNorm, epsilon=1e-5)

        layers: list[nnx.Module] = []

        layers.append(
            nnx.Sequential(
                nnx.Conv(
                    3,
                    embed_dim,
                    kernel_size=(patch_size[0], patch_size[1]),
                    strides=(patch_size[0], patch_size[1]),
                    rngs=rngs,
                ),
                norm_layer(embed_dim, rngs=rngs),
            ),
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0

        for i_stage in range(len(depths)):
            stage: list[nnx.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        rngs=rngs,
                    ),
                )
                stage_block_id += 1
            layers.append(nnx.Sequential(*stage))

            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer, rngs=rngs))
        self.features = nnx.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features, rngs=rngs)
        self.head = nnx.Linear(num_features, num_classes, rngs=rngs)

        for _, m in self.iter_modules():
            if isinstance(m, nnx.Linear):
                m.kernel_init = nnx.initializers.truncated_normal(stddev=0.02)
                if m.bias is not None:
                    m.bias_init = nnx.initializers.zeros_init()

    def __call__(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = x.mean(axis=(1, 2))
        return self.head(x)


def _swin_transformer(
    patch_size: list[int],
    embed_dim: int,
    depths: list[int],
    num_heads: list[int],
    window_size: list[int],
    stochastic_depth_prob: float,
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> SwinTransformer:
    return SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        rngs=rngs,
        **kwargs,
    )


def swin_t(*, rngs: nnx.Rngs, **kwargs: Any) -> SwinTransformer:
    """Constructs a swin_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_T_Weights
        :members:

    """
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        rngs=rngs,
        **kwargs,
    )


def swin_s(*, rngs: nnx.Rngs, **kwargs: Any) -> SwinTransformer:
    """Constructs a swin_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_S_Weights
        :members:

    """
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        rngs=rngs,
        **kwargs,
    )


def swin_b(*, rngs: nnx.Rngs, **kwargs: Any) -> SwinTransformer:
    """Constructs a swin_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_B_Weights
        :members:

    """
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        rngs=rngs,
        **kwargs,
    )


def swin_v2_t(*, rngs: nnx.Rngs, **kwargs: Any) -> SwinTransformer:
    """Constructs a swin_v2_tiny architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_T_Weights
        :members:

    """
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        rngs=rngs,
        **kwargs,
    )


def swin_v2_s(*, rngs: nnx.Rngs, **kwargs: Any) -> SwinTransformer:
    """Constructs a swin_v2_small architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_S_Weights
        :members:

    """
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        rngs=rngs,
        **kwargs,
    )


def swin_v2_b(*, rngs: nnx.Rngs, **kwargs: Any) -> SwinTransformer:
    """Constructs a swin_v2_base architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_B_Weights
        :members:

    """
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        rngs=rngs,
        **kwargs,
    )
