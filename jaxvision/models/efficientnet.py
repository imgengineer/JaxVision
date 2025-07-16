import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..ops.stochastic_depth import StochasticDepth
from ._utils import _make_divisible

__all__ = [
    "EfficientNet",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_l",
    "efficientnet_v2_m",
    "efficientnet_v2_s",
]


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nnx.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: int | None = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Callable[..., nnx.Module] | None = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return math.ceil(num_layers * depth_mult)


class FusedMBConvConfig(_MBConvConfig):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Callable[..., nnx.Module] | None = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class MBConv(nnx.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nnx.Module],
        se_layer: Callable[..., nnx.Module] = SqueezeExcitation,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        if not (1 <= cnf.stride <= 2):
            msg = "illegal stride value"
            raise ValueError(msg)

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nnx.Module] = []
        activation_layer = nnx.silu

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    rngs=rngs,
                ),
            )

        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                rngs=rngs,
            ),
        )

        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=nnx.silu,
                rngs=rngs,
            ),
        )

        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
                rngs=rngs,
            ),
        )

        self.deterministic = deterministic
        self.block = nnx.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(rate=stochastic_depth_prob, mode="row", rngs=rngs)
        self.out_channels = cnf.out_channels

    def __call__(self, inputs: jax.Array) -> jax.Array:
        result = self.block(inputs)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += inputs
        return result


class FusedMBConv(nnx.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nnx.Module],
        *,
        rngs: nnx.Rngs,
        deterministic: bool = False,
    ) -> None:
        if not (1 <= cnf.stride <= 2):
            msg = "illegal stride value"
            raise ValueError(msg)

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nnx.Module] = []
        activation_layer = nnx.silu

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    rngs=rngs,
                ),
            )

            layers.append(
                Conv2dNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                    rngs=rngs,
                ),
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    rngs=rngs,
                ),
            )
        self.deterministic = deterministic
        self.block = nnx.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(rate=stochastic_depth_prob, mode="row", rngs=rngs)
        self.out_channels = cnf.out_channels

    def __call__(self, inputs: jax.Array) -> jax.Array:
        result = self.block(inputs)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += inputs
        return result


class EfficientNet(nnx.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Callable[..., nnx.Module] | None = None,
        last_channel: int | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """EfficientNet V1 and V2 main class.

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer

        """
        if not inverted_residual_setting:
            msg = "The inverted_residual_setting should not be empty"
            raise ValueError(msg)
        if not (isinstance(inverted_residual_setting, Sequence) and all(isinstance(s, _MBConvConfig) for s in inverted_residual_setting)):
            msg = "The inverted_residual_setting should be List[MBConvConfig]"
            raise TypeError(msg)
        if norm_layer is None:
            norm_layer = nnx.BatchNorm

        layers: list[nnx.Module] = []

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nnx.silu,
                rngs=rngs,
            ),
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: list[nnx.Module] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)

                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer, rngs=rngs))
                stage_block_id += 1

            layers.append(nnx.Sequential(*stage))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nnx.silu,
                rngs=rngs,
            ),
        )

        self.features = nnx.Sequential(*layers)
        self.classifier = nnx.Sequential(
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(lastconv_output_channels, num_classes, rngs=rngs),
        )

        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv):
                m.kernel_init = nnx.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")
                if m.bias is not None:
                    m.bias_init = nnx.initializers.zeros_init()
            elif isinstance(m, nnx.BatchNorm | nnx.GroupNorm):
                m.scale_init = nnx.initializers.ones_init()
                m.bias_init = nnx.initializers.zeros_init()
            elif isinstance(m, nnx.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                m.kernel_init = nnx.initializers.truncated_normal(lower=-init_range, upper=init_range)
                m.bias_init = nnx.initializers.zeros_init()

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.features(x)
        x = jnp.mean(x, axis=(1, 2))
        return self.classifier(x)


def _efficientnet(
    inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig],
    dropout: float,
    last_channel: int | None,
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    return EfficientNet(
        inverted_residual_setting,
        dropout,
        last_channel=last_channel,
        rngs=rngs,
        **kwargs,
    )


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> tuple[Sequence[MBConvConfig | FusedMBConvConfig], int | None]:
    inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        msg = f"Unsupported model type {arch}"
        raise ValueError(msg)

    return inverted_residual_setting, last_channel


def efficientnet_b0(
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> EfficientNet:
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        rngs=rngs,
        **kwargs,
    )


def efficientnet_b1(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B1_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        rngs=rngs,
        **kwargs,
    )


def efficientnet_b2(*, rngs: nnx.Rngs, **kwargs: Any) -> EfficientNet:
    """EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B2_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        rngs=rngs,
        **kwargs,
    )


def efficientnet_b3(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B3_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B3_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        rngs=rngs,
        **kwargs,
    )


def efficientnet_b4(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B4_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B4_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B4_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        rngs=rngs,
        **kwargs,
    )


def efficientnet_b5(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B5_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B5_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B5_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        rngs=rngs,
        norm_layer=partial(nnx.BatchNorm, epsilon=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b6(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B6 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B6_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B6_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B6_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.5),
        last_channel,
        rngs=rngs,
        norm_layer=partial(nnx.BatchNorm, epsilon=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b7(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B7 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B7_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B7_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B7_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.5),
        last_channel,
        rngs=rngs,
        norm_layer=partial(nnx.BatchNorm, epsilon=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_v2_s(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_S_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        rngs=rngs,
        norm_layer=partial(nnx.BatchNorm, epsilon=1e-03),
        **kwargs,
    )


def efficientnet_v2_m(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """Constructs an EfficientNetV2-M architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_M_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_M_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_M_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        rngs=rngs,
        norm_layer=partial(nnx.BatchNorm, epsilon=1e-03),
        **kwargs,
    )


def efficientnet_v2_l(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> EfficientNet:
    """Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_L_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_L_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_L_Weights
        :members:

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        rngs=rngs,
        norm_layer=partial(nnx.BatchNorm, epsilon=1e-03),
        **kwargs,
    )
