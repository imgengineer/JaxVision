from flax import nnx
from typing import Optional, Union, Tuple, Callable, List
from jax import Array
import jax.numpy as jnp


class Conv2dNormActivation(nnx.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nnx.Module]] = nnx.BatchNorm,
        activation_layer: Optional[Callable[..., nnx.Module]] = nnx.relu,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., nnx.Module] = nnx.Conv,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = "SAME"
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=padding,
                kernel_dilation=dilation,
                feature_group_count=groups,
                use_bias=bias,
                rngs=rngs,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels, rngs=rngs))

        if activation_layer is not None:
            layers.append(activation_layer)

        self.out_channels = out_channels
        super().__init__(*layers)


class SqueezeExtraction(nnx.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nnx.Module] = nnx.relu,
        scale_activation: Callable[..., nnx.Module] = nnx.sigmoid,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.fc1 = nnx.Conv(
            input_channels, squeeze_channels, kernel_size=(1, 1), rngs=rngs
        )
        self.fc2 = nnx.Conv(
            squeeze_channels, input_channels, kernel_size=(1, 1), rngs=rngs
        )
        self.activation = activation
        self.scale_activation = scale_activation

    def _scale(self, input: Array) -> Array:
        scale = jnp.mean(input, axis=(1, 2), keepdims=True)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def __call__(self, input: Array) -> Array:
        scale = self._scale(input)
        return scale * input


class MLP(nnx.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        drop
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., nnx.Module]] = None,
        activation_layer: Optional[Callable[..., nnx.Module]] = nnx.relu,
        bias: bool = False,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nnx.Linear(in_dim, hidden_dim, use_bias=bias, rngs=rngs))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim, rngs=rngs))
            layers.append(activation_layer)
            layers.append(nnx.Dropout(rate=dropout, rngs=rngs))
            in_dim = hidden_dim
        layers.append(nnx.Linear(in_dim, hidden_channels[-1], use_bias=bias, rngs=rngs))
        layers.append(nnx.Dropout(rate=dropout, rngs=rngs))

        super().__init__(*layers)
