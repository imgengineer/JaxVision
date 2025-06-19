import math
from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp
from flax import nnx
from jax import Array

from ..ops.misc import Conv2dNormActivation, ReLU, SqueezeExtraction
from ._utils import _make_divisible

__all__ = [
    "RegNet",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_x_8gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
    "regnet_y_8gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
    "regnet_y_128gf",
    "regnet_y_400mf",
    "regnet_y_800mf",
]


class SimpleStemIN(Conv2dNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nnx.Module],
        activation_layer: Callable[..., nnx.Module],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            rngs=rngs,
        )


class BottleneckTransform(nnx.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(  # noqa: PLR0913
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nnx.Module],
        activation_layer: Callable[..., nnx.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float | None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        layers: list[nnx.Module] = []
        w_b = round(width_out * bottleneck_multiplier)
        g = w_b // group_width

        layers.append(
            Conv2dNormActivation(
                width_in,
                w_b,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                rngs=rngs,
            ),
        )

        layers.append(
            Conv2dNormActivation(
                w_b,
                w_b,
                kernel_size=3,
                stride=stride,
                groups=g,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                rngs=rngs,
            ),
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = round(se_ratio * width_in)
            layers.append(
                SqueezeExtraction(
                    input_channels=w_b,
                    squeeze_channels=width_se_out,
                    activation=activation_layer,
                    rngs=rngs,
                ),
            )

        layers.append(
            Conv2dNormActivation(
                w_b,
                width_out,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=None,
                rngs=rngs,
            ),
        )
        super().__init__(*layers)


class ResBottlenectBlock(nnx.Module):
    """Residual bottleneck block: x + F(x), F = bottlenect transform."""

    def __init__(  # noqa: PLR0913
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nnx.Module],
        activation_layer: Callable[..., nnx.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: float | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = Conv2dNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
                rngs=rngs,
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
            rngs=rngs,
        )
        self.activation = activation_layer()

    def __call__(self, x: Array) -> Array:
        x = self.proj(x) + self.f(x) if self.proj is not None else x + self.f(x)
        return self.activation(x)


class AnyStage(nnx.Sequential):
    """AnyNet stage (sequence of blocks w/ the sample output shape)."""

    def __init__(  # noqa: PLR0913
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nnx.Module],
        norm_layer: Callable[..., nnx.Module],
        activation_layer: Callable[..., nnx.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        layers: list[nnx.Module] = []
        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
                rngs=rngs,
            )
            layers.append(block)
        super().__init__(*layers)


class BlockParams:
    def __init__(  # noqa: PLR0913
        self,
        depths: list[int],
        widths: list[int],
        group_widths: list[int],
        bottleneck_multipliers: list[float],
        strides: list[int],
        se_ratio: float | None = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(  # noqa: PLR0913
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: float | None = None,
    ) -> "BlockParams":
        """Programmatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """
        QUANT = 8  # noqa: N806
        STRIDE = 2  # noqa: N806

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            msg = "Invalid RegNet settings"
            raise ValueError(msg)
        # Compute the block widths. Each stage has one unique block width
        widths_cont = jnp.arange(depth) * w_a + w_0
        block_capacity = jnp.round(jnp.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (
            (jnp.round(jnp.divide(w_0 * jnp.pow(w_m, block_capacity), QUANT)) * QUANT).astype(jnp.int32).tolist()
        )

        num_stages = len(set(block_widths))

        # Convert to per stage paramsters
        split_helper = zip(
            [*block_widths, 0],
            [0, *block_widths],
            [*block_widths, 0],
            [0, *block_widths],
            strict=False,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1], strict=False) if t]
        stage_depths = jnp.diff(jnp.array([d for d, t in enumerate(splits) if t])).astype(jnp.int32).tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibility(
            stage_widths,
            bottleneck_multipliers,
            group_widths,
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
            strict=False,
        )

    @staticmethod
    def _adjust_widths_groups_compatibility(
        stage_widths: list[int],
        bottleneck_ratios: list[float],
        group_widths: list[int],
    ) -> tuple[list[int], list[int]]:
        """Adjust the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios, strict=False)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths, strict=False)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min, strict=False)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios, strict=False)]
        return stage_widths, group_widths_min


class RegNet(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Callable[..., nnx.Module] | None = None,
        block_type: Callable[..., nnx.Module] | None = None,
        norm_layer: Callable[..., nnx.Module] | None = None,
        activation: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nnx.BatchNorm
        if block_type is None:
            block_type = ResBottlenectBlock
        if activation is None:
            activation = ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
            rngs=rngs,
        )

        current_width = stem_width

        blocks = []
        for _i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):  # noqa: SLF001
            blocks.append(
                AnyStage(
                    current_width,
                    width_out,
                    stride,
                    depth,
                    block_type,
                    norm_layer,
                    activation,
                    group_width,
                    bottleneck_multiplier,
                    block_params.se_ratio,
                    rngs=rngs,
                ),
            )
            current_width = width_out

        self.trunk_output = nnx.Sequential(*blocks)

        self.fc = nnx.Linear(current_width, num_classes, rngs=rngs)

        # Performs ResNet-style weight initialization
        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv):
                fan_out = m.kernel.value.shape[0] * m.kernel.value.shape[1] * m.out_features
                m.kernel_init = nnx.initializers.normal(stddev=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nnx.BatchNorm):
                m.scale_init = nnx.initializers.ones_init()
                m.bias_init = nnx.initializers.zeros_init()
            elif isinstance(m, nnx.Linear):
                m.kernel_init = nnx.initializers.normal(stddev=0.01)
                m.bias_init = nnx.initializers.zeros_init()

    def __call__(self, x: Array) -> Array:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = x.mean(axis=(1, 2))
        return self.fc(x)


def _regnet(block_params: BlockParams, *, rngs: nnx.Rngs, **kwargs):
    norm_layer = kwargs.pop("norm_layer", partial(nnx.BatchNorm, epsilon=1e-05, momentum=0.1))
    return RegNet(block_params, norm_layer=norm_layer, rngs=rngs, **kwargs)


def regnet_y_400mf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetY_400MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_400MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_400MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_400MF_Weights
        :members:

    """
    params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25)
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_y_800mf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetY_800MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_800MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_800MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_800MF_Weights
        :members:

    """
    params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25)
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_y_1_6gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetY_1.6GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_1_6GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_1_6GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_1_6GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=27,
        w_0=48,
        w_a=20.71,
        w_m=2.65,
        group_width=24,
        se_ratio=0.25,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_y_3_2gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetY_3.2GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_3_2GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_3_2GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_3_2GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=21,
        w_0=80,
        w_a=42.63,
        w_m=2.66,
        group_width=24,
        se_ratio=0.25,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_y_8gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetY_8GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_8GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_8GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_8GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=17,
        w_0=192,
        w_a=76.82,
        w_m=2.19,
        group_width=56,
        se_ratio=0.25,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_y_16gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetY_16GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_16GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_16GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_16GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=18,
        w_0=200,
        w_a=106.23,
        w_m=2.48,
        group_width=112,
        se_ratio=0.25,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_y_32gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetY_32GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_32GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_32GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_32GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=20,
        w_0=232,
        w_a=115.89,
        w_m=2.53,
        group_width=232,
        se_ratio=0.25,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_y_128gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetY_128GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_128GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_128GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_128GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=27,
        w_0=456,
        w_a=160.83,
        w_m=2.52,
        group_width=264,
        se_ratio=0.25,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_x_400mf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetX_400MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_400MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_400MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_400MF_Weights
        :members:

    """
    params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16)
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_x_800mf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetX_800MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_800MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_800MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_800MF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=16,
        w_0=56,
        w_a=35.73,
        w_m=2.28,
        group_width=16,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_x_1_6gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetX_1.6GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_1_6GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_1_6GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_1_6GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=18,
        w_0=80,
        w_a=34.01,
        w_m=2.25,
        group_width=24,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_x_3_2gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetX_3.2GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_3_2GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_3_2GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_3_2GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=25,
        w_0=88,
        w_a=26.31,
        w_m=2.25,
        group_width=48,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_x_8gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetX_8GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_8GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_8GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_8GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=23,
        w_0=80,
        w_a=49.56,
        w_m=2.88,
        group_width=120,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_x_16gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetX_16GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_16GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_16GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_16GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=22,
        w_0=216,
        w_a=55.59,
        w_m=2.1,
        group_width=128,
    )
    return _regnet(params, rngs=rngs, **kwargs)


def regnet_x_32gf(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> RegNet:
    """Constructs a RegNetX_32GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_32GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_32GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_32GF_Weights
        :members:

    """
    params = BlockParams.from_init_params(
        depth=23,
        w_0=320,
        w_a=69.86,
        w_m=2.0,
        group_width=168,
    )
    return _regnet(params, rngs=rngs, **kwargs)
