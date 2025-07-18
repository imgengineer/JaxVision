import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..ops.misc import Conv2dNormActivation, Identity, SqueezeExcitation
from ..ops.stochastic_depth import StochasticDepth

__all__ = [
    "MaxVit",
    "maxvit_t",
]


def _get_conv_output_shape(input_size: tuple[int, int], kernel_size: int, stride: int, padding: int) -> tuple[int, int]:
    return (
        (input_size[0] - kernel_size + 2 * padding) // stride + 1,
        (input_size[1] - kernel_size + 2 * padding) // stride + 1,
    )


def _make_block_input_shapes(input_size: tuple[int, int], n_blocks: int) -> list[tuple[int, int]]:
    """Util function to check that the input size is correct for a MaxVit configuration."""
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes


def _get_relative_position_index(height: int, width: int) -> jax.Array:
    coords_h = jnp.arange(height)
    coords_w = jnp.arange(width)
    coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flat = coords.reshape(coords.shape[0], -1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.transpose(1, 2, 0)
    relative_coords_h = relative_coords[:, :, 0] + (height - 1)
    relative_coords_w = relative_coords[:, :, 1] + (width - 1)
    return relative_coords_h * (2 * width - 1) + relative_coords_w


class MBConv(nnx.Module):
    """MBConv: Mobile Inverted Residual Bottleneck.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        p_stochastic_dropout (float): Probability of stochastic depth.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        squeeze_ratio: float,
        stride: int,
        activation_layer: Callable[..., nnx.Module],
        norm_layer: Callable[..., nnx.Module],
        p_stochastic_dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        proj: Sequence[nnx.Module]
        self.proj: nnx.Module

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), strides=(1, 1), use_bias=True, rngs=rngs)]
            if stride == 2:
                proj = [partial(nnx.avg_pool, window_shape=(3, 3), strides=(stride, stride), padding="SAME"), *proj]
            self.proj = nnx.Sequential(*proj)
        else:
            self.proj = Identity()

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.stochastic_depath = StochasticDepth(p_stochastic_dropout, mode="row", rngs=rngs)
        else:
            self.stochastic_depath = Identity()
        _layers = []
        _layers.extend(
            [
                norm_layer(in_channels, rngs=rngs),
                Conv2dNormActivation(
                    in_channels,
                    mid_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    rngs=rngs,
                ),
                Conv2dNormActivation(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    groups=mid_channels,
                    rngs=rngs,
                ),
                SqueezeExcitation(mid_channels, sqz_channels, activation=nnx.silu, rngs=rngs),
                nnx.Conv(
                    mid_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    use_bias=True,
                    rngs=rngs,
                ),
            ],
        )
        self.layers = nnx.Sequential(*_layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Args:
            x (jax.Array): Input tensor with expected layout of [B, H, W, C].

        Returns:
            jax.Array: Output tensor with expected layout of [B, H / stride, W / stride, C].

        """
        res = self.proj(x)
        x = self.stochastic_depath(self.layers(x))
        return res + x


class RelativePositionMultiHeadAttention(nnx.Module):
    """Relative Positional Multi-Head Attention.

    Args:
        feat_dim (int): Number of input features.
        head_dim (int): Number of features per head.
        max_seq_len (int): Maximum sequence length.

    """

    def __init__(
        self,
        feat_dim: int,
        head_dim: int,
        max_seq_len: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if feat_dim % head_dim != 0:
            msg = f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}"
            raise ValueError(msg)

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nnx.Linear(feat_dim, self.n_heads * self.head_dim * 3, rngs=rngs)
        self.scale_factor = feat_dim**-0.5

        self.merge = nnx.Linear(self.head_dim * self.n_heads, feat_dim, rngs=rngs)
        self.relative_position_bias_table = nnx.Param(
            nnx.initializers.zeros_init()(rngs.params(), ((2 * self.size - 1) * (2 * self.size - 1), self.n_heads)),
        )
        self.relative_position_index = _get_relative_position_index(self.size, self.size)

        self.relative_position_bias_table.value = nnx.initializers.truncated_normal(stddev=0.02)(
            rngs.params(),
            self.relative_position_bias_table.value.shape,
        )

    def get_relative_positional_bias(self) -> jax.Array:
        bias_index = self.relative_position_index.reshape(-1)
        relative_bias = self.relative_position_bias_table[bias_index].reshape(
            self.max_seq_len,
            self.max_seq_len,
            -1,
        )
        relative_bias = relative_bias.transpose(2, 0, 1)
        return jnp.expand_dims(relative_bias, axis=0)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Args:
            x (jax.Array): Input tensor with expected layout of [B, G, P, D].

        Returns:
            jax.Array: Output tensor with expected layout of [B, G, P, D].

        """
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, G, P, H, DH).transpose(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).transpose(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).transpose(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = jnp.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = nnx.softmax(dot_prod + pos_bias, axis=-1)

        out = jnp.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.transpose(0, 1, 3, 2, 4).reshape(B, G, P, D)

        return self.merge(out)


class SwapAxes(nnx.Module):
    """Permute the axes of a tensor."""

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def __call__(self, x: jax.Array) -> jax.Array:
        return x.swapaxes(self.a, self.b)


class WindowPartition(nnx.Module):
    """Partition the input tensor into non-overlapping windows."""

    def __call__(self, x: jax.Array, p: int) -> jax.Array:
        """Args:
            x (jax.Array): Input tensor with expected layout of [B, H, W, C].
            p (int): Number of partitions.

        Returns:
            : Output tensor with expected layout of [B, H/P, W/P, P*P, C].

        """
        B, H, W, C = x.shape
        P = p

        x = x.reshape(B, H // P, P, W // P, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, (H // P) * (W // P), P * P, C)


class WindowDepartition(nnx.Module):
    """Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W]."""

    def __call__(self, x: jax.Array, p: int, h_partitions: int, w_partitions: int) -> jax.Array:
        """Args:
            x (jax.Array): Input tensor with expected layout of [B, (H/P * W/P), P*P, C].
            p (int): Number of partitions.
            h_partitions (int): Number of vertical partitions.
            w_partitions (int): Number of horizontal partitions.

        Returns:
            jax.Array: Output tensor with expected layout of [B, H, W, C].

        """
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions

        x = x.reshape(B, HP, WP, P, P, C)

        x = x.transpose(0, 1, 3, 2, 4, 5)

        return x.reshape(B, HP * P, WP * P, C)


class PartitionAttentionLayer(nnx.Module):
    """Layer for partitioning the input tensor into non-overlapping windows and applying attention to each window.

    Args:
        in_channels (int): Number of input channels.
        head_dim (int): Dimension of each attention head.
        partition_size (int): Size of the partitions.
        partition_type (str): Type of partitioning to use. Can be either "grid" or "window".
        grid_size (tuple[int, int]): Size of the grid to partition the input tensor into.
        mlp_ratio (int): Ratio of the  feature size expansion in the MLP layer.
        activation_layer (Callable[..., nn.Module]): Activation function to use.
        norm_layer (Callable[..., nn.Module]): Normalization function to use.
        attention_dropout (float): Dropout probability for the attention layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        p_stochastic_dropout (float): Probability of dropping out a partition.

    """

    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        partition_size: int,
        partition_type: str,
        grid_size: tuple[int, int],
        mlp_ratio: int,
        activation_layer: Callable[..., nnx.Module],
        norm_layer: Callable[..., nnx.Module],
        attention_dropout: float,
        mlp_dropout: float,
        p_stochastic_dropout: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.partition_type = partition_type
        self.grid_size = grid_size

        if partition_type not in ["grid", "window"]:
            msg = "partition_type must be either 'grid' or 'window'"
            raise ValueError(msg)

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxes(-2, -3) if partition_type == "grid" else Identity()
        self.departition_swap = SwapAxes(-2, -3) if partition_type == "grid" else Identity()

        self.attn_layer = nnx.Sequential(
            norm_layer(in_channels, rngs=rngs),
            RelativePositionMultiHeadAttention(in_channels, head_dim, partition_size**2, rngs=rngs),
            nnx.Dropout(attention_dropout, rngs=rngs),
        )
        self.mlp_layer = nnx.Sequential(
            nnx.LayerNorm(in_channels, rngs=rngs),
            nnx.Linear(in_channels, in_channels * mlp_ratio, rngs=rngs),
            activation_layer,
            nnx.Linear(in_channels * mlp_ratio, in_channels, rngs=rngs),
            nnx.Dropout(mlp_dropout, rngs=rngs),
        )

        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row", rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Args:
            x (jax.Array): Input tensor with expected layout of [B, H, W, C].

        Returns:
            jax.Array: Output tensor with expected layout of [B, H, W, C].

        """
        gh, gw = self.grid_size[0] // self.p, self.grid_size[1] // self.p
        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.stochastic_dropout(self.attn_layer(x))
        x = x + self.stochastic_dropout(self.mlp_layer(x))
        x = self.departition_swap(x)
        return self.departition_op(x, self.p, gh, gw)


class MaxVitLayer(nnx.Module):
    """MaxVit layer consisting of a MBConv layer followed by a PartitionAttentionLayer with `window` and a PartitionAttentionLayer with `grid`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        grid_size (tuple[int, int]): Size of the input feature grid.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        stride: int,
        norm_layer: Callable[..., nnx.Module],
        activation_layer: Callable[..., nnx.Module],
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        p_stochastic_dropout: float,
        partition_size: int,
        grid_size: tuple[int, int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        layers = []
        layers.extend(
            [
                MBConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion_ratio=expansion_ratio,
                    squeeze_ratio=squeeze_ratio,
                    stride=stride,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    p_stochastic_dropout=p_stochastic_dropout,
                    rngs=rngs,
                ),
                PartitionAttentionLayer(
                    in_channels=out_channels,
                    head_dim=head_dim,
                    partition_size=partition_size,
                    partition_type="window",
                    grid_size=grid_size,
                    mlp_ratio=mlp_ratio,
                    activation_layer=activation_layer,
                    norm_layer=nnx.LayerNorm,
                    attention_dropout=attention_dropout,
                    mlp_dropout=mlp_dropout,
                    p_stochastic_dropout=p_stochastic_dropout,
                    rngs=rngs,
                ),
                PartitionAttentionLayer(
                    in_channels=out_channels,
                    head_dim=head_dim,
                    partition_size=partition_size,
                    partition_type="grid",
                    grid_size=grid_size,
                    mlp_ratio=mlp_ratio,
                    activation_layer=activation_layer,
                    norm_layer=nnx.LayerNorm,
                    attention_dropout=attention_dropout,
                    mlp_dropout=mlp_dropout,
                    p_stochastic_dropout=p_stochastic_dropout,
                    rngs=rngs,
                ),
            ],
        )
        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Args:
            x (jax.Array): Input tensor of shape (B, H, W, C).

        Returns:
            jax.Array: Output tensor of shape (B, H, W, C).

        """
        return self.layers(x)


class MaxVitBlock(nnx.Module):
    """A MaxVit block consisting of `n_layers` MaxVit layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        input_grid_size (tuple[int, int]): Size of the input feature grid.
        n_layers (int): Number of layers in the block.
        p_stochastic (list[float]): list of probabilities for stochastic depth for each layer.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        norm_layer: Callable[..., nnx.Module],
        activation_layer: Callable[..., nnx.Module],
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        partition_size: int,
        input_grid_size: tuple[int, int],
        n_layers: int,
        p_stochastic: list[float],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if len(p_stochastic) != n_layers:
            msg = f"p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}."
            raise ValueError(msg)

        self.grid_size = _get_conv_output_shape(input_grid_size, kernel_size=3, stride=2, padding=1)

        layers = []
        for idx, p in enumerate(p_stochastic):
            stride = 2 if idx == 0 else 1
            layers.append(
                MaxVitLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    grid_size=self.grid_size,
                    p_stochastic_dropout=p,
                    rngs=rngs,
                ),
            )
        self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x


class MaxVit(nnx.Module):
    """Implements MaxVit Transformer from the `MaxViT: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`_ paper.

    Args:
        input_size (tuple[int, int]): Size of the input image.
        stem_channels (int): Number of channels in the stem.
        partition_size (int): Size of the partitions.
        block_channels (list[int]): Number of channels in each block.
        block_layers (list[int]): Number of layers in each block.
        stochastic_depth_prob (float): Probability of stochastic depth. Expands to a list of probabilities for each layer that scales linearly to the specified value.
        squeeze_ratio (float): Squeeze ratio in the SE Layer. Default: 0.25.
        expansion_ratio (float): Expansion ratio in the MBConv bottleneck. Default: 4.
        norm_layer (Callable[..., nn.Module]): Normalization function. Default: None (setting to None will produce a `BatchNorm2d(eps=1e-3, momentum=0.01)`).
        activation_layer (Callable[..., nn.Module]): Activation function Default: nn.GELU.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Expansion ratio of the MLP layer. Default: 4.
        mlp_dropout (float): Dropout probability for the MLP layer. Default: 0.0.
        attention_dropout (float): Dropout probability for the attention layer. Default: 0.0.
        num_classes (int): Number of classes. Default: 1000.

    """

    def __init__(
        self,
        input_size: tuple[int, int],
        stem_channels: int,
        partition_size: int,
        block_channels: list[int],
        block_layers: list[int],
        head_dim: int,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nnx.Module] | None = None,
        activation_layer: Callable[..., nnx.Module] = nnx.gelu,
        squeeze_ratio: float = 0.25,
        expansion_ratio: float = 4,
        mlp_ratio: int = 4,
        mlp_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        input_channels = 3

        if norm_layer is None:
            norm_layer = partial(nnx.BatchNorm, epsilon=1e-3, momentum=0.01)
        block_input_sizes = _make_block_input_shapes(input_size, len(block_channels))
        for idx, block_input_size in enumerate(block_input_sizes):
            if block_input_size[0] % partition_size != 0 or block_input_size[1] % partition_size != 0:
                msg = (
                    f"Input size {block_input_size} of block {idx} is not divisible by partition size {partition_size}. "
                    f"Consider changing the partition size or the input size.\n"
                    f"Current configuration yields the following block input sizes: {block_input_sizes}."
                )
                raise ValueError(
                    msg,
                )

        self.stem = nnx.Sequential(
            Conv2dNormActivation(
                input_channels,
                stem_channels,
                3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                bias=False,
                rngs=rngs,
            ),
            Conv2dNormActivation(
                stem_channels,
                stem_channels,
                3,
                stride=1,
                norm_layer=None,
                activation_layer=None,
                bias=True,
                rngs=rngs,
            ),
        )

        input_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        self.partition_size = partition_size

        blocks = []
        in_channels = [stem_channels, *block_channels[:-1]]
        out_channels = block_channels

        p_stochastic = np.linspace(0, stochastic_depth_prob, sum(block_layers)).tolist()

        p_idx = 0
        for in_channel, out_channel, num_layers in zip(in_channels, out_channels, block_layers, strict=True):
            blocks.append(
                MaxVitBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    input_grid_size=input_size,
                    n_layers=num_layers,
                    p_stochastic=p_stochastic[p_idx : p_idx + num_layers],
                    rngs=rngs,
                ),
            )
            input_size = blocks[-1].grid_size
            p_idx += num_layers

        self.blocks = blocks
        self.classifier = nnx.Sequential(
            nnx.LayerNorm(block_channels[-1], rngs=rngs),
            nnx.Linear(block_channels[-1], block_channels[-1], rngs=rngs),
            nnx.tanh,
            nnx.Linear(block_channels[-1], num_classes, use_bias=False, rngs=rngs),
        )

        self._init_weights()

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(axis=(1, 2))
        return self.classifier(x)

    def _init_weights(self):
        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv):
                m.kernel_init = nnx.initializers.normal(stddev=0.02)
                if m.bias is not None:
                    m.bias_init = nnx.initializers.zeros_init()
                elif isinstance(m, nnx.BatchNorm):
                    m.scale_init = nnx.initializers.constant(1)
                    m.bias_init = nnx.initializers.constant(0)
                elif isinstance(m, nnx.Linear):
                    m.kernel_init = nnx.initializers.normal(stddev=0.02)
                    if m.bias is not None:
                        m.bias_init = nnx.initializers.zeros_init()


def _maxvit(
    stem_channels: int,
    block_channels: list[int],
    block_layers: list[int],
    stochastic_depth_prob: float,
    partition_size: int,
    head_dim: int,
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> MaxVit:
    input_size = kwargs.pop("input_size", (224, 224))

    return MaxVit(
        stem_channels=stem_channels,
        block_channels=block_channels,
        block_layers=block_layers,
        stochastic_depth_prob=stochastic_depth_prob,
        head_dim=head_dim,
        partition_size=partition_size,
        input_size=input_size,
        rngs=rngs,
        **kwargs,
    )


def maxvit_t(*, rngs: nnx.Rngs, **kwargs: Any) -> MaxVit:
    """Constructs a maxvit_t architecture from
    `MaxViT: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`_.

    Args:
        weights (:class:`~torchvision.models.MaxVit_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MaxVit_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.maxvit.MaxVit``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/maxvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MaxVit_T_Weights
        :members:

    """
    return _maxvit(
        stem_channels=64,
        block_channels=[64, 128, 256, 512],
        block_layers=[2, 2, 5, 2],
        head_dim=32,
        stochastic_depth_prob=0.2,
        partition_size=7,
        rngs=rngs,
        **kwargs,
    )
