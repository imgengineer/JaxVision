from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax.numpy as jnp
from flax import nnx
from jax import Array

from ops.misc import Conv2dNormActivation
from ops.stochastic_depth import StochasticDepth

__all__ = [
    "ConvNeXt",
    "convnext_base",
    "convnext_large",
    "convnext_small",
    "convnext_tiny",
]


class CNBlock(nnx.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nnx.LayerNorm, epsilon=1e-6)

        self.block = nnx.Sequential(
            nnx.Conv(dim, dim, kernel_size=(7, 7), padding="SAME", feature_group_count=dim, use_bias=True, rngs=rngs),
            norm_layer(dim, rngs=rngs),
            nnx.Linear(in_features=dim, out_features=4 * dim, use_bias=True, rngs=rngs),
            nnx.gelu,
            nnx.Linear(in_features=4 * dim, out_features=dim, use_bias=True, rngs=rngs),
        )
        self.layer_scale = nnx.Param(jnp.full(shape=(1, 1, 1, dim), fill_value=layer_scale))
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row", rngs=rngs)

    def __call__(self, inputs: Array) -> Array:
        result = self.layer_scale * self.block(inputs)
        result = self.stochastic_depth(result)
        result += inputs
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: int | None,
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        block_setting: list[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Callable[..., nnx.Module] | None = None,
        norm_layer: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all(isinstance(s, CNBlockConfig) for s in block_setting)):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(nnx.LayerNorm, epsilon=1e-6)

        layers: list[nnx.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
                rngs=rngs,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: list[nnx.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob, rngs=rngs))
                stage_block_id += 1
            layers.append(nnx.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nnx.Sequential(
                        norm_layer(cnf.input_channels, rngs=rngs),
                        nnx.Conv(cnf.input_channels, cnf.out_channels, kernel_size=(2, 2), strides=(2, 2), rngs=rngs),
                    )
                )

        self.features = nnx.Sequential(*layers)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nnx.Sequential(
            norm_layer(lastconv_output_channels, rngs=rngs),
            nnx.Linear(lastconv_output_channels, num_classes, rngs=rngs),
        )

    def __call__(self, x: Array) -> Array:
        x = self.features(x)
        x = x.mean(axis=(1, 2))
        x = self.classifier(x)
        return x


def _convnext(
    block_setting: list[CNBlockConfig],
    stochastic_depth_prob: float,
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> ConvNeXt:
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, rngs=rngs, **kwargs)

    return model


def convnext_tiny(*, rngs: nnx.Rngs, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, rngs=rngs, **kwargs)


def convnext_small(*, rngs: nnx.Rngs, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Small model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Small_Weights
        :members:
    """

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(block_setting, stochastic_depth_prob, rngs=rngs, **kwargs)


def convnext_base(*, rngs: nnx.Rngs, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Base model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Base_Weights
        :members:
    """

    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, rngs=rngs, **kwargs)


def convnext_large(*, rngs: nnx.Rngs, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Large model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Large_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, rngs=rngs, **kwargs)
