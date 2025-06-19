from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from flax import nnx
from jax import Array

__all__ = [
    "ShuffleNetV2",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


def channel_shuffle(x: Array, groups: int) -> Array:
    bacth_size, height, width, num_channels = x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = x.reshape(bacth_size, height, width, groups, channels_per_group)

    x = jnp.transpose(x, (0, 1, 2, 4, 3))

    # flatten
    return x.reshape(bacth_size, height, width, num_channels)


class InvertedResidual(nnx.Module):
    def __init__(self, inp: int, oup: int, stride: int, *, rngs: nnx.Rngs):
        super().__init__()

        if not (1 <= stride <= 3):  # noqa: PLR2004
            msg = "illegal stride value"
            raise ValueError(msg)
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            msg = f"Invalid combination of stride {stride},inp {inp} and oup {oup} values. If stride == 1 the inp should be equal to oup // 2 << 1"  # noqa: E501
            raise ValueError(msg)

        if self.stride > 1:
            self.branch1 = nnx.Sequential(
                self.depthwise_conv(
                    inp,
                    inp,
                    kernel_size=(3, 3),
                    stride=self.stride,
                    padding=(1, 1),
                    rngs=rngs,
                ),
                nnx.BatchNorm(inp, rngs=rngs),
                nnx.Conv(
                    inp,
                    branch_features,
                    kernel_size=(1, 1),
                    strides=1,
                    padding="SAME",
                    use_bias=False,
                    rngs=rngs,
                ),
                nnx.BatchNorm(branch_features, rngs=rngs),
                nnx.relu,
            )
        else:
            self.branch1 = nnx.Sequential()

        self.branch2 = nnx.Sequential(
            nnx.Conv(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=(1, 1),
                strides=1,
                padding=(0, 0),
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(branch_features, rngs=rngs),
            nnx.relu,
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=(3, 3),
                stride=self.stride,
                padding=(1, 1),
                rngs=rngs,
            ),
            nnx.BatchNorm(branch_features, rngs=rngs),
            nnx.Conv(
                branch_features,
                branch_features,
                kernel_size=(1, 1),
                strides=1,
                padding=(0, 0),
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(branch_features, rngs=rngs),
            nnx.relu,
        )

    @staticmethod
    def depthwise_conv(  # noqa: PLR0913
        i: int,
        o: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: tuple[int, int] = (0, 0),
        *,
        bias: bool = False,
        rngs: nnx.Rngs,
    ) -> nnx.Conv:
        return nnx.Conv(i, o, kernel_size, stride, padding=padding, use_bias=bias, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        if self.stride == 1:
            x1, x2 = jnp.split(x, 2, axis=3)
            out = jnp.concat([x1, self.branch2(x2)], axis=3)
        else:
            out = jnp.concat([self.branch1(x), self.branch2(x)], axis=3)

        return channel_shuffle(out, 2)


class ShuffleNetV2(nnx.Module):
    def __init__(
        self,
        stages_repeats: list[int],
        stages_out_channels: list[int],
        num_classes: int = 1000,
        inverted_residual: Callable[..., nnx.Module] = InvertedResidual,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        if len(stages_repeats) != 3:  # noqa: PLR2004
            msg = "expected stages_repeats as list of 3 positive ints"
            raise ValueError(msg)
        if len(stages_out_channels) != 5:  # noqa: PLR2004
            msg = "expected stages_out_channels as list of 5 positive ints"
            raise ValueError(msg)
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nnx.Sequential(
            nnx.Conv(
                input_channels,
                output_channels,
                kernel_size=(3, 3),
                strides=2,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(output_channels, rngs=rngs),
            nnx.relu,
        )
        input_channels = output_channels

        self.maxpool = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        # Static annotations for mypy
        self.stage2: nnx.Sequential
        self.stage3: nnx.Sequential
        self.stage4: nnx.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names,
            stages_repeats,
            self._stage_out_channels[1:],
            strict=False,
        ):
            seq = [inverted_residual(input_channels, output_channels, 2, rngs=rngs)]
            seq.extend([inverted_residual(output_channels, output_channels, 1, rngs=rngs) for _ in range(repeats - 1)])
            setattr(self, name, nnx.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nnx.Sequential(
            nnx.Conv(
                input_channels,
                output_channels,
                kernel_size=(1, 1),
                strides=1,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(output_channels, rngs=rngs),
            nnx.relu,
        )

        self.fc = nnx.Linear(output_channels, num_classes, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean(axis=(1, 2))
        return self.fc(x)


def _shufflenetv2(*, rngs: nnx.Rngs, **kwargs) -> ShuffleNetV2:
    return ShuffleNetV2(rngs=rngs, **kwargs)


def shufflenet_v2_x0_5(*, rngs: nnx.Rngs, **kwargs):
    return _shufflenetv2(
        rngs=rngs,
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 48, 96, 192, 1024],
        **kwargs,
    )


def shufflenet_v2_x1_0(*, rngs: nnx.Rngs, **kwargs):
    return _shufflenetv2(
        rngs=rngs,
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 116, 232, 464, 1024],
        **kwargs,
    )


def shufflenet_v2_x1_5(*, rngs: nnx.Rngs, **kwargs):
    return _shufflenetv2(
        rngs=rngs,
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 176, 352, 704, 1024],
        **kwargs,
    )


def shufflenet_v2_x2_0(*, rngs: nnx.Rngs, **kwargs):
    return _shufflenetv2(
        rngs=rngs,
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 244, 488, 976, 2048],
        **kwargs,
    )
