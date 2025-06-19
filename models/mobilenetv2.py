from collections.abc import Callable

from flax import nnx
from jax import Array

from ops.misc import Conv2dNormActivation, ReLU6

from ._utils import _make_divisible

__all__ = ["MobileNetV2", "mobilenet_v2"]


class InvertedResidual(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            msg = f"stride shoule be 1 or 2 instead {stride}"
            raise ValueError(msg)

        if norm_layer is None:
            norm_layer = nnx.BatchNorm

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: list[nnx.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=ReLU6,
                    rngs=rngs,
                ),
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=ReLU6,
                    rngs=rngs,
                ),
                # pw-linear
                nnx.Conv(
                    hidden_dim,
                    oup,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="SAME",
                    use_bias=False,
                    rngs=rngs,
                ),
                norm_layer(oup, rngs=rngs),
            ],
        )
        self.conv = nnx.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def __call__(self, x: Array) -> Array:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: list[list[int]] | None = None,
        round_nearest: int = 8,
        block: Callable[..., nnx.Module] | None = None,
        norm_layer: Callable[..., nnx.Module] | None = None,
        dropout: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """MobileNet V2 main class.

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nnx.BatchNorm

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user known t, c, n, s are required
        if (
            len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4  # noqa: PLR2004
        ):
            msg = f"inveted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            raise ValueError(msg)

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: list[nnx.Module] = [
            Conv2dNormActivation(
                3,
                input_channel,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=ReLU6,
                rngs=rngs,
            ),
        ]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                        rngs=rngs,
                    ),
                )
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=ReLU6,
                rngs=rngs,
            ),
        )

        # make it nnx.Sequential
        self.features = nnx.Sequential(*features)

        # building classifier
        self.classifier = nnx.Sequential(
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(self.last_channel, num_classes, rngs=rngs),
        )

        # weight initialization
        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv):
                m.kernel_init = nnx.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")
                if m.bias is not None:
                    m.bias_init = nnx.initializers.zeros_init()
            elif isinstance(m, nnx.BatchNorm | nnx.GroupNorm):
                m.scale_init = nnx.initializers.ones_init()
                m.bias_init = nnx.initializers.zeros_init()
            elif isinstance(m, nnx.Linear):
                m.kernel_init = nnx.initializers.normal(stddev=0.01)
                m.bias_init = nnx.initializers.zeros_init()

    def __call__(self, x: Array) -> Array:
        x = self.features(x)
        x = x.mean(axis=(1, 2))
        return self.classifier(x)


def mobilenet_v2(*, rngs: nnx.Rngs, **kwargs) -> MobileNetV2:
    return MobileNetV2(rngs=rngs, **kwargs)
