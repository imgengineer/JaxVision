from functools import partial
from typing import Any

import jax.numpy as jnp
from flax import nnx
from jax import Array

__all__ = [
    "DenseNet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]


class _DenseLayer(nnx.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm1 = nnx.BatchNorm(num_input_features, rngs=rngs)
        self.conv1 = nnx.Conv(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

        self.norm2 = nnx.BatchNorm(bn_size * growth_rate, rngs=rngs)
        self.conv2 = nnx.Conv(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

        self.drop_rate = float(drop_rate)
        self.dropout = nnx.Dropout(rate=self.drop_rate, rngs=rngs)

    def bn_function(self, inputs: list[Array]) -> Array:
        concated_features = jnp.concat(inputs, axis=3)
        return self.conv1(nnx.relu(self.norm1(concated_features)))

    def __call__(self, inputs: Array) -> Array:
        prev_features = [inputs] if isinstance(inputs, Array) else inputs

        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(nnx.relu(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = self.dropout(new_features)

        return new_features


class _DenseBlock(nnx.Module):
    _version = 2

    def __init__(  # noqa: PLR0913
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        layers = [
            _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                rngs=rngs,
            )
            for i in range(num_layers)
        ]
        self.layers = layers

    def __call__(self, init_features: Array) -> Array:
        features = [init_features]
        for layer in self.layers:
            new_features = layer(jnp.concat(features, axis=-1))
            features.append(new_features)
        return jnp.concat(features, axis=-1)


class _Transition(nnx.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int, *, rngs: nnx.Rngs) -> None:
        layers = [
            nnx.BatchNorm(num_input_features, rngs=rngs),
            nnx.relu,
            nnx.Conv(
                num_input_features,
                num_output_features,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=False,
                rngs=rngs,
            ),
            partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2)),
        ]
        super().__init__(*layers)


class DenseNet(nnx.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes

    """

    def __init__(  # noqa: PLR0913
        self,
        growth_rate: int = 32,
        block_config: tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        *,
        rngs: nnx.Rngs,
    ) -> None:

        features = [
            nnx.Conv(
                3,
                num_init_features,
                kernel_size=(7, 7),
                strides=(2, 2),
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(num_init_features, rngs=rngs),
            nnx.relu,
            partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME"),
        ]

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                rngs=rngs,
            )
            features.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    rngs=rngs,
                )
                features.append(trans)
                num_features = num_features // 2
        features.append(nnx.BatchNorm(num_features, rngs=rngs))

        self.features = nnx.Sequential(*features)

        self.classifier = nnx.Linear(num_features, num_classes, rngs=rngs)

        # Official init from torch repo
        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv):
                m.kernel_init = nnx.initializers.kaiming_normal()
            elif isinstance(m, nnx.BatchNorm):
                m.scale_init = nnx.initializers.constant(1)
                m.bias_init = nnx.initializers.constant(0)
            elif isinstance(m, nnx.Linear):
                m.bias_init = nnx.initializers.constant(0)

    def __call__(self, x: Array) -> Array:
        features = self.features(x)
        out = nnx.relu(features)
        out = out.mean(axis=(1, 2))
        return self.classifier(out)


def _densenet(
    growth_rate: int,
    block_config: tuple[int, int, int, int],
    num_init_features: int,
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> DenseNet:
    return DenseNet(growth_rate, block_config, num_init_features, rngs=rngs, **kwargs)


def densenet121(*, rngs: nnx.Rngs, **kwargs: Any) -> DenseNet:
    return _densenet(32, (6, 12, 24, 16), 64, rngs=rngs, **kwargs)


def densenet161(*, rngs: nnx.Rngs, **kwargs: Any) -> DenseNet:
    return _densenet(48, (6, 12, 36, 24), 96, rngs=rngs, **kwargs)


def densenet169(*, rngs: nnx.Rngs, **kwargs) -> DenseNet:
    return _densenet(32, (6, 12, 32, 32), 64, rngs=rngs, **kwargs)


def densenet201(*, rngs: nnx.Rngs, **kwargs) -> DenseNet:
    return _densenet(32, (6, 12, 48, 32), 64, rngs=rngs, **kwargs)
