import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import first_from
from jax import Array


class SqueezeExtraction(nnx.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nnx.Module] = nnx.relu,
        scale_activation: Callable[..., nnx.Module] = nnx.sigmoid,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.fc1 = nnx.Conv(input_channels, squeeze_channels, kernel_size=(1, 1), rngs=rngs)
        self.fc2 = nnx.Conv(squeeze_channels, input_channels, kernel_size=(1, 1), rngs=rngs)
        self.activation = activation
        self.scale_activation = scale_activation

    def _scale(self, x: Array) -> Array:
        scale = x.mean(axis=(1, 2), keepdims=True)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def __call__(self, inputs: Array) -> Array:
        scale = self._scale(inputs)
        return scale * inputs


class MLP(nnx.Sequential):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        hidden_channels: list[int],
        norm_layer: Callable[..., nnx.Module] | None = None,
        activation_layer: Callable[..., nnx.Module] | None = nnx.relu,
        dropout: float = 0.0,
        *,
        bias: bool = False,
        rngs: nnx.Rngs,
    ) -> nnx.Sequential:
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


class Identity(nnx.Module):
    def __call__(self, inputs: Array) -> Array:
        return inputs


@dataclasses.dataclass(repr=False)
class DropPath(nnx.Module):
    drop_prob: float = 0.0
    scale_by_keep: bool = True
    deterministic: bool = False
    rng_collection: str = "drop_path"
    rngs: rnglib.Rngs | None = None

    def __call__(self, inputs: Array, *, deterministic: bool | None = None, rngs: rnglib.Rngs | None = None) -> Array:
        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to DropPath
            as either a __call__ argument or class attribute""",
        )
        if (self.drop_prob == 0.0) or deterministic:
            return inputs
        # Prevent gradient NaNs in 1.0 edge-case.
        if self.drop_prob == 1.0:
            return jnp.zeros_like(inputs)

        keep_prob = 1.0 - self.drop_prob
        rng = rngs[self.rng_collection]() if rngs else self.rngs[self.rng_collection]()
        broadcast_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, inputs.shape)

        return jax.lax.select(mask, inputs / keep_prob if self.scale_by_keep else inputs, jnp.zeros_like(inputs))


class Conv2dNormActivation(nnx.Sequential):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | tuple[int, int] | str | None = None,
        groups: int = 1,
        norm_layer: Callable[..., nnx.Module] | None = nnx.BatchNorm,
        activation_layer: Callable[..., nnx.Module] | None = nnx.relu,
        dilation: int | tuple[int, ...] = 1,
        conv_layer: Callable[..., nnx.Module] = nnx.Conv,
        *,
        bias: bool | None = None,
        rngs: nnx.Rngs,
    ) -> nnx.Sequential:
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
            ),
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels, rngs=rngs))

        if activation_layer is not None:
            layers.append(activation_layer)

        self.out_channels = out_channels
        super().__init__(*layers)
