from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from ..ops.misc import DropPath


class StemLayer(nnx.Module):
    """Stem layer for initial feature extraction.
    Code adapted from InternImage: https://github.com/OpenGVLab/InternImage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer=nnx.gelu,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),
        *,
        rngs: nnx.Rngs,
    ):


        self.conv1 = nnx.Conv(
            in_channels,
            out_channels // 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.norm1 = norm_layer(out_channels // 2, rngs=rngs)
        self.act = act_layer


        self.conv2 = nnx.Conv(
            out_channels // 2,
            out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.norm2 = norm_layer(out_channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Performs the forward pass of the StemLayer.

        Args:
            x: Input feature map (N, H, W, C).

        Returns:
            Output feature map after stem operations.

        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        return self.norm2(x)


class DownsampleLayer(nnx.Module):
    """Downsampling layer for feature map reduction.
    Code adapted from InternImage: https://github.com/OpenGVLab/InternImage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),
        *,
        rngs: nnx.Rngs,
    ):


        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.norm = norm_layer(out_channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Performs the forward pass of the DownsampleLayer.

        Args:
            x: Input feature map (N, H, W, C).

        Returns:
            Output feature map after downsampling.

        """
        x = self.conv(x)
        return self.norm(x)


class MlpHead(nnx.Module):
    """MLP classification head."""

    def __init__(
        self,
        dim: int,
        num_classes: int = 1000,
        act_layer=nnx.gelu,
        mlp_ratio: int = 4,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),
        head_dropout: float = 0.0,
        *,
        bias: bool = True,
        rngs: nnx.Rngs,
    ):

        hidden_features = int(mlp_ratio * dim)


        self.fc1 = nnx.Linear(dim, hidden_features, use_bias=bias, rngs=rngs)
        self.act = act_layer
        self.norm = norm_layer(hidden_features, rngs=rngs)


        self.fc2 = nnx.Linear(hidden_features, num_classes, use_bias=bias, rngs=rngs)
        self.head_dropout = nnx.Dropout(rate=head_dropout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Performs the forward pass of the MlpHead.

        Args:
            x: Input features.

        Returns:
            Output logits.

        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        return self.fc2(x)


class GatedCNNBlock(nnx.Module):
    """Implementation of Gated CNN Block.

    Args:
        conv_ratio: controls the number of channels to conduct depthwise convolution.
                    Conduct convolution on partial channels can improve practical efficiency.
                    The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
                    also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)

    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 8 / 3,
        kernel_size: int = 7,
        conv_ratio: float = 1.0,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),
        act_layer=nnx.gelu,
        drop_path: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm = norm_layer(dim, rngs=rngs)


        hidden = int(expansion_ratio * dim)

        self.fc1 = nnx.Linear(dim, hidden * 2, rngs=rngs)
        self.act = act_layer


        conv_channels = int(conv_ratio * dim)


        size_g = hidden
        size_i = hidden - conv_channels
        size_c = conv_channels



        if size_g < 0 or size_i < 0 or size_c < 0:
            msg = (
                f"Calculated split sizes must be non-negative. "
                f"Ensure expansion_ratio ({expansion_ratio}) * dim ({dim}) >= conv_ratio ({conv_ratio}) * dim ({dim}). "
                f"Got: g={size_g}, i={size_i}, c={size_c}"
            )
            raise ValueError(msg)





        self.split_points = [size_g, size_g + size_i]



        self.conv = nnx.Conv(
            conv_channels,
            conv_channels,
            kernel_size=(kernel_size, kernel_size),
            padding="SAME",
            feature_group_count=conv_channels,
            rngs=rngs,
        )

        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)


        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path, rngs=rngs)
        else:
            self.drop_path = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Performs the forward pass of the GatedCNNBlock.

        Args:
            x: Input features (N, H, W, C).

        Returns:
            Output features after block operations.

        """
        shortcut = x
        x = self.norm(x)

        fc1_output = self.fc1(x)


        g, i, c = jnp.split(fc1_output, self.split_points, axis=-1)



        c = self.conv(c)



        x = self.fc2(self.act(g) * jnp.concatenate([i, c], axis=-1))


        if self.drop_path is not None:
            x = self.drop_path(x)


        return x + shortcut




DOWNSAMPLE_LAYERS_FOUR_STAGES = [StemLayer] + [DownsampleLayer] * 3


class MambaOut(nnx.Module):
    """MetaFormer - MambaOut
    A Flax NNX implementation of: `MetaFormer Baselines for Vision` - https://arxiv.org/abs/2210.13452.

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [3, 3, 9, 3].
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 576].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        output_norm: norm before classifier head. Default: partial(nnx.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: MlpHead.
        head_dropout (float): dropout for MLP classifier. Default: 0.

    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),
        act_layer=nnx.gelu,
        conv_ratio: float = 1.0,
        kernel_size: int = 7,
        drop_path_rate: float = 0.0,
        output_norm=partial(nnx.LayerNorm, epsilon=1e-6),
        head_fn=MlpHead,
        head_dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):

        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 576]

        self.num_classes = num_classes


        if not isinstance(depths, list | tuple):
            depths = [depths]
        if not isinstance(dims, list | tuple):
            dims = [dims]

        self.num_stages = len(depths)


        if not isinstance(downsample_layers, list | tuple):
            downsample_layers = [downsample_layers] * self.num_stages


        down_dims = [in_chans, *dims]

        self.downsample_layers = [
            downsample_layers[i](in_channels=down_dims[i], out_channels=down_dims[i + 1], rngs=rngs)
            for i in range(self.num_stages)
        ]



        dp_rates = jnp.linspace(0.0, drop_path_rate, sum(depths)).tolist()


        self.stages = []
        cur_dp_idx = 0
        for i in range(self.num_stages):
            stage_blocks = [
                GatedCNNBlock(
                    dim=dims[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    drop_path=dp_rates[cur_dp_idx + j],
                    rngs=rngs,
                )
                for j in range(depths[i])
            ]
            self.stages.append(stage_blocks)
            cur_dp_idx += depths[i]


        self.norm = output_norm(dims[-1], rngs=rngs)


        self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout, rngs=rngs)

        self._init_weights()

    def _init_weights(self):
        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv | nnx.Linear):
                m.kernel_init = nnx.initializers.truncated_normal(stddev=0.02)
                if m.bias is not None:
                    m.bias_init = nnx.initializers.constant(0)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Performs the full forward pass of the MambaOut model.

        Args:
            x: Input image (N, H, W, C).

        Returns:
            Output logits.

        """
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)

            for block in self.stages[i]:
                x = block(x)


        x = self.norm(x.mean(axis=(1, 2)))
        return self.head(x)



def mambaout_femto(*, rngs: nnx.Rngs, **kwargs) -> MambaOut:
    """Creates a MambaOut-Femto model.

    Args:
        rngs: Random number generators for NNX modules.
        **kwargs: Additional keyword arguments for MambaOut constructor.

    Returns:
        An instance of MambaOut-Femto model.

    """
    return MambaOut(
        depths=[3, 3, 9, 3],
        dims=[48, 96, 192, 288],
        rngs=rngs,
        **kwargs,
    )


def mambaout_kobe(*, rngs: nnx.Rngs, **kwargs) -> MambaOut:
    """Creates a MambaOut-Kobe model (Kobe Memorial Version with 24 Gated CNN blocks).

    Args:
        rngs: Random number generators for NNX modules.
        **kwargs: Additional keyword arguments for MambaOut constructor.

    Returns:
        An instance of MambaOut-Kobe model.

    """
    return MambaOut(
        depths=[3, 3, 15, 3],
        dims=[48, 96, 192, 288],
        rngs=rngs,
        **kwargs,
    )


def mambaout_tiny(*, rngs: nnx.Rngs, **kwargs) -> MambaOut:
    """Creates a MambaOut-Tiny model.

    Args:
        rngs: Random number generators for NNX modules.
        **kwargs: Additional keyword arguments for MambaOut constructor.

    Returns:
        An instance of MambaOut-Tiny model.

    """
    return MambaOut(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 576],
        rngs=rngs,
        **kwargs,
    )


def mambaout_small(*, rngs: nnx.Rngs, **kwargs) -> MambaOut:
    """Creates a MambaOut-Small model.

    Args:
        rngs: Random number generators for NNX modules.
        **kwargs: Additional keyword arguments for MambaOut constructor.

    Returns:
        An instance of MambaOut-Small model.

    """
    return MambaOut(
        depths=[3, 4, 27, 3],
        dims=[96, 192, 384, 576],
        rngs=rngs,
        **kwargs,
    )


def mambaout_base(*, rngs: nnx.Rngs, **kwargs) -> MambaOut:
    """Creates a MambaOut-Base model.

    Args:
        rngs: Random number generators for NNX modules.
        **kwargs: Additional keyword arguments for MambaOut constructor.

    Returns:
        An instance of MambaOut-Base model.

    """
    return MambaOut(
        depths=[3, 4, 27, 3],
        dims=[128, 256, 512, 768],
        rngs=rngs,
        **kwargs,
    )
