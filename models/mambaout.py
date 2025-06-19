from functools import partial

import jax.numpy as jnp
from flax import nnx
from jax import Array

from ops.misc import DropPath


class StemLayer(nnx.Module):
    """Stem layer for initial feature extraction.
    Code adapted from InternImage: https://github.com/OpenGVLab/InternImage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer=nnx.gelu,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        # JAX Conv expects input format (N, H, W, C)
        # Convolution 1: Halves spatial dimensions, halves channels (out_channels // 2)
        self.conv1 = nnx.Conv(
            in_channels, out_channels // 2, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.norm1 = norm_layer(out_channels // 2, rngs=rngs)  # Normalize along the last dimension (channels)
        self.act = act_layer  # Activation function (e.g., GELU)

        # Convolution 2: Halves spatial dimensions again, doubles channels back to out_channels
        self.conv2 = nnx.Conv(
            out_channels // 2,
            out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.norm2 = norm_layer(out_channels, rngs=rngs)  # Normalize along the last dimension (channels)

    def __call__(self, x: Array) -> Array:
        """Performs the forward pass of the StemLayer.

        Args:
            x: Input feature map (N, H, W, C).

        Returns:
            Output feature map after stem operations.

        """
        # Input is already in (N, H, W, C) format, apply convolutions and normalizations
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
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        # JAX Conv expects input format (N, H, W, C)
        # Convolution to halve spatial dimensions and change channels
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.norm = norm_layer(out_channels, rngs=rngs)  # Normalize along the last dimension (channels)

    def __call__(self, x: Array) -> Array:
        """Performs the forward pass of the DownsampleLayer.

        Args:
            x: Input feature map (N, H, W, C).

        Returns:
            Output feature map after downsampling.

        """
        # Input is already in (N, H, W, C) format, apply convolution directly
        x = self.conv(x)
        return self.norm(x)


class MlpHead(nnx.Module):
    """MLP classification head."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_classes: int = 1000,
        act_layer=nnx.gelu,
        mlp_ratio: int = 4,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        head_dropout: float = 0.0,
        *,
        bias: bool = True,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        # Calculate hidden features dimension
        hidden_features = int(mlp_ratio * dim)

        # Linear layer 1
        self.fc1 = nnx.Linear(dim, hidden_features, use_bias=bias, rngs=rngs)
        self.act = act_layer  # Activation function
        self.norm = norm_layer(hidden_features, rngs=rngs)  # Normalization layer

        # Linear layer 2 (output to num_classes)
        self.fc2 = nnx.Linear(hidden_features, num_classes, use_bias=bias, rngs=rngs)
        self.head_dropout = nnx.Dropout(rate=head_dropout, rngs=rngs)  # Dropout layer

    def __call__(self, x: Array) -> Array:
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

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        expansion_ratio: float = 8 / 3,
        kernel_size: int = 7,
        conv_ratio: float = 1.0,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        act_layer=nnx.gelu,
        drop_path: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.norm = norm_layer(dim, rngs=rngs)

        # Calculate hidden features dimension
        hidden = int(expansion_ratio * dim)
        # First linear layer to expand features
        self.fc1 = nnx.Linear(dim, hidden * 2, rngs=rngs)
        self.act = act_layer

        # Calculate channels for depthwise convolution
        conv_channels = int(conv_ratio * dim)

        # Calculate the sizes of the three segments
        size_g = hidden
        size_i = hidden - conv_channels
        size_c = conv_channels

        # Validate sizes to ensure they are non-negative.
        # This is a safeguard against invalid expansion_ratio or conv_ratio combinations.
        if size_g < 0 or size_i < 0 or size_c < 0:
            msg = (
                f"Calculated split sizes must be non-negative. "
                f"Ensure expansion_ratio ({expansion_ratio}) * dim ({dim}) >= conv_ratio ({conv_ratio}) * dim ({dim}). "
                f"Got: g={size_g}, i={size_i}, c={size_c}"
            )
            raise ValueError(msg)

        # Define the split points for jnp.split.
        # jnp.split with a list of integers splits *at* those indices.
        # To get segments of size size_g, size_i, size_c, the split points are:
        # [size_g, size_g + size_i]
        self.split_points = [size_g, size_g + size_i]

        # Depthwise Convolution: feature_group_count = conv_channels makes it depthwise
        # JAX Conv expects input (N, H, W, C), so no permutation needed around conv here
        self.conv = nnx.Conv(
            conv_channels,
            conv_channels,
            kernel_size=(kernel_size, kernel_size),
            padding="SAME",
            feature_group_count=conv_channels,  # This creates a depthwise convolution
            rngs=rngs,
        )
        # Second linear layer to project features back to original dimension
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)

        # DropPath layer
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path, rngs=rngs)
        else:
            self.drop_path = None

    def __call__(self, x: Array) -> Array:
        """Performs the forward pass of the GatedCNNBlock.

        Args:
            x: Input features (N, H, W, C).

        Returns:
            Output features after block operations.

        """
        shortcut = x  # Save shortcut for residual connection
        x = self.norm(x)  # Apply normalization

        fc1_output = self.fc1(x)

        # Use self.split_points which are the indices where to split
        g, i, c = jnp.split(fc1_output, self.split_points, axis=-1)

        # Apply depthwise convolution to 'c'
        # 'c' is already in (N, H, W, C) format, so nnx.Conv can be applied directly.
        c = self.conv(c)

        # Concatenate 'i' and 'c' along the last dimension
        # Apply activation to 'g' and multiply with the concatenated features
        x = self.fc2(self.act(g) * jnp.concatenate([i, c], axis=-1))

        # Apply DropPath if enabled
        if self.drop_path is not None:
            x = self.drop_path(x)

        # Add shortcut for residual connection
        return x + shortcut


# Define downsampling layers for four stages
# First stage uses StemLayer, subsequent three stages use DownsampleLayer
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

    def __init__(  # noqa: PLR0913
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        act_layer=nnx.gelu,
        conv_ratio: float = 1.0,
        kernel_size: int = 7,
        drop_path_rate: float = 0.0,
        output_norm=partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        head_fn=MlpHead,
        head_dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        # Set default depths and dimensions if not provided
        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 576]

        self.num_classes = num_classes

        # Ensure depths and dims are lists for consistent iteration
        if not isinstance(depths, list | tuple):
            depths = [depths]
        if not isinstance(dims, list | tuple):
            dims = [dims]

        self.num_stages = len(depths)  # Number of stages in the model

        # Ensure downsample_layers is a list
        if not isinstance(downsample_layers, list | tuple):
            downsample_layers = [downsample_layers] * self.num_stages

        # Prepare dimensions for downsampling layers
        down_dims = [in_chans, *dims]
        # Instantiate downsampling layers
        self.downsample_layers = [
            downsample_layers[i](in_channels=down_dims[i], out_channels=down_dims[i + 1], rngs=rngs)
            for i in range(self.num_stages)
        ]

        # Calculate individual drop path rates for each block
        # JAX's jnp.linspace is used here instead of torch.linspace
        dp_rates = jnp.linspace(0.0, drop_path_rate, sum(depths)).tolist()

        # Instantiate stages, each containing a list of GatedCNNBlocks
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
                    drop_path=dp_rates[cur_dp_idx + j],  # Assign per-block drop path rate
                    rngs=rngs,
                )
                for j in range(depths[i])
            ]
            self.stages.append(stage_blocks)
            cur_dp_idx += depths[i]

        # Output normalization layer
        self.norm = output_norm(dims[-1], rngs=rngs)

        # Classification head
        self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout, rngs=rngs)

        self._init_weights()

    def _init_weights(self):
        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv | nnx.Linear):
                m.kernel_init = nnx.initializers.truncated_normal(stddev=0.02)
                if m.bias is not None:
                    m.bias_init = nnx.initializers.constant(0)

    def __call__(self, x: Array) -> Array:
        """Performs the full forward pass of the MambaOut model.

        Args:
            x: Input image (N, H, W, C).

        Returns:
            Output logits.

        """
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)  # Apply downsampling
            # Apply all blocks in the current stage
            for block in self.stages[i]:
                x = block(x)

        # Then apply output normalization
        x = self.norm(x.mean(axis=(1, 2)))  # Extract features
        return self.head(x)  # Pass to classification head


# Helper function to create MambaOut models
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
