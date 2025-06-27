from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ..ops.misc import Conv2dNormActivation
from ..ops.misc import SqueezeExcitation as SElayer
from ._utils import _make_divisible

__all__ = [
    "MobileNetV3",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
]


class InvertedResidualConfig:
    # Stores information listed at Table 1 and 2 of the MobileNetV3 paper
    def __init__(  # noqa: PLR0913
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,  # noqa: FBT001
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertResidual(nnx.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nnx.Module],
        se_layer: Callable[..., nnx.Module] = partial(SElayer, scale_activation=nnx.hard_sigmoid),  # noqa: B008
        *,
        rngs: nnx.Rngs,
    ):
        if not (1 <= cnf.stride <= 2):  # noqa: PLR2004
            msg = "illegal stride value"
            raise ValueError(msg)

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nnx.Module] = []
        activation_layer = nnx.hard_sigmoid if cnf.use_hs else nnx.relu

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    rngs=rngs,
                ),
            )
        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                rngs=rngs,
            ),
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels, rngs=rngs))

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
                rngs=rngs,
            ),
        )

        self.block = nnx.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def __call__(self, input: jax.Array) -> jax.Array:  # noqa: A002
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        inverted_residual_setting: list[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Callable[..., nnx.Module] | None = None,
        norm_layer: Callable[..., nnx.Module] | None = None,
        dropout: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """MobileNet V3 main class.

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        if not inverted_residual_setting:
            msg = "The inverted_residual_setting should not be empty"
            raise ValueError(msg)
        if not (
            isinstance(inverted_residual_setting, Sequence)
            and all(isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting)
        ):
            msg = "The inverted_residual_setting should be List[InvertedResidualConfig]"
            raise TypeError(msg)

        if block is None:
            block = InvertResidual

        if norm_layer is None:
            norm_layer = partial(nnx.BatchNorm, epsilon=0.001, momentum=0.01)

        layers: list[nnx.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nnx.hard_swish,
                rngs=rngs,
            ),
        )

        # building inverted residual blocks
        layers.extend([block(cnf, norm_layer, rngs=rngs) for cnf in inverted_residual_setting])

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nnx.hard_swish,
                rngs=rngs,
            ),
        )

        self.features = nnx.Sequential(*layers)
        self.classifier = nnx.Sequential(
            nnx.Linear(lastconv_output_channels, last_channel, rngs=rngs),
            nnx.hard_swish,
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(last_channel, num_classes, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.features(x)
        x = jnp.mean(x, axis=(1, 2))
        return self.classifier(x)


def _mobilenet_v3_conf(
    arch: str,
    width_mult: float = 1.0,
    *,
    reduced_tail: bool = False,
    dilated: bool = False,
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(
                112,
                5,
                672,
                160 // reduce_divider,
                True,
                "HS",
                2,
                dilation,
            ),  # C4
            bneck_conf(
                160 // reduce_divider,
                5,
                960 // reduce_divider,
                160 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
            bneck_conf(
                160 // reduce_divider,
                5,
                960 // reduce_divider,
                160 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(
                96 // reduce_divider,
                5,
                576 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
            bneck_conf(
                96 // reduce_divider,
                5,
                576 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        msg = f"Unsupported model type {arch}"
        raise ValueError(msg)

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    inverted_residual_setting: list[InvertedResidualConfig],
    last_channel: int,
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> MobileNetV3:
    return MobileNetV3(inverted_residual_setting, last_channel, rngs=rngs, **kwargs)


def mobilenet_v3_large(*, rngs: nnx.Rngs, **kwargs) -> MobileNetV3:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large",
    )
    return _mobilenet_v3(inverted_residual_setting, last_channel, rngs=rngs, **kwargs)


def mobilenet_v3_small(*, rngs: nnx.Rngs, **kwargs) -> MobileNetV3:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_small",
    )
    return _mobilenet_v3(inverted_residual_setting, last_channel, rngs=rngs, **kwargs)
