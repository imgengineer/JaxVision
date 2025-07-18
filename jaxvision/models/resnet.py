from collections.abc import Callable
from functools import partial
from typing import Any

import jax
from flax import nnx

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    *,
    rngs: nnx.Rngs,
) -> nnx.Conv:
    """3x3 convolution with padding."""
    return nnx.Conv(
        in_planes,
        out_planes,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="SAME",
        feature_group_count=groups,
        use_bias=False,
        kernel_dilation=(dilation, dilation),
        rngs=rngs,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, *, rngs: nnx.Rngs) -> nnx.Conv:
    """1x1 convolution."""
    return nnx.Conv(
        in_planes,
        out_planes,
        kernel_size=(1, 1),
        strides=(stride, stride),
        use_bias=False,
        rngs=rngs,
    )


class BasicBlock(nnx.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nnx.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if norm_layer is None:
            norm_layer = nnx.BatchNorm
        if groups != 1 or base_width != 64:
            msg = "BasicBlock only supports groups=1 and base_width=64"
            raise ValueError(msg)
        if dilation > 1:
            msg = "Dilation > 1 not supported in BasicBlock"
            raise NotImplementedError(msg)

        self.conv1 = conv3x3(inplanes, planes, stride, rngs=rngs)
        self.bn1 = norm_layer(planes, rngs=rngs)
        self.conv2 = conv3x3(planes, planes, rngs=rngs)
        self.bn2 = norm_layer(planes, rngs=rngs)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: jax.Array) -> jax.Array:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return nnx.relu(out)


class Bottleneck(nnx.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nnx.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if norm_layer is None:
            norm_layer = nnx.BatchNorm
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width, rngs=rngs)
        self.bn1 = norm_layer(width, rngs=rngs)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, rngs=rngs)
        self.bn2 = norm_layer(width, rngs=rngs)
        self.conv3 = conv1x1(width, planes * self.expansion, rngs=rngs)
        self.bn3 = norm_layer(planes * self.expansion, rngs=rngs)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: jax.Array) -> jax.Array:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nnx.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return nnx.relu(out)


class ResNet(nnx.Module):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nnx.Module] | None = None,
        *,
        zero_init_residual: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        if norm_layer is None:
            norm_layer = nnx.BatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            msg = f"replace_stride_with_dilation shoule be None or a 3-element tuple, got {replace_stride_with_dilation}"
            raise ValueError(msg)
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nnx.Conv(
            3,
            self.inplanes,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = norm_layer(self.inplanes, rngs=rngs)
        self.maxpool = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        self.layer1 = self._make_layer(block, 64, layers[0], rngs=rngs)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            rngs=rngs,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            rngs=rngs,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            rngs=rngs,
        )
        self.fc = nnx.Linear(512 * block.expansion, num_classes, rngs=rngs)

        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv):
                m.kernel_init = nnx.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")
            elif isinstance(m, nnx.BatchNorm | nnx.GroupNorm):
                m.scale_init = nnx.initializers.constant(1)
                m.bias_init = nnx.initializers.constant(0)

        if zero_init_residual:
            for _, m in self.iter_modules():
                if isinstance(m, Bottleneck) and m.bn3.scale is not None:
                    m.bn3.scale_init = nnx.initializers.constant(0)
                elif isinstance(m, BasicBlock) and m.bn2.scale is not None:
                    m.bn2.scale_init = nnx.initializers.constant(0)

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        *,
        dilate: bool = False,
        rngs: nnx.Rngs,
    ) -> nnx.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nnx.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, rngs=rngs),
                norm_layer(planes * block.expansion, rngs=rngs),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                rngs=rngs,
            ),
        )
        self.inplanes = planes * block.expansion
        layers.extend(
            [
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    rngs=rngs,
                )
                for _ in range(1, blocks)
            ],
        )
        return nnx.Sequential(*layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(axis=(1, 2))
        return self.fc(x)


def _resnet(
    block: type[BasicBlock | Bottleneck],
    layers: list[int],
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> ResNet:
    return ResNet(block, layers, rngs=rngs, **kwargs)


def resnet18(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], rngs=rngs, **kwargs)


def resnet34(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], rngs=rngs, **kwargs)


def resnet50(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], rngs=rngs, **kwargs)


def resnet101(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], rngs=rngs, **kwargs)


def resnet152(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], rngs=rngs, **kwargs)


def resnext50_32x4d(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:

    """
    return _resnet(Bottleneck, [3, 4, 6, 3], rngs=rngs, groups=32, width_per_group=4, **kwargs)


def resnext101_32x8d(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:

    """
    return _resnet(Bottleneck, [3, 4, 23, 3], rngs=rngs, groups=32, width_per_group=8, **kwargs)


def resnext101_64x4d(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:

    """
    return _resnet(Bottleneck, [3, 4, 23, 3], rngs=rngs, groups=64, width_per_group=4, **kwargs)


def wide_resnet50_2(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:

    """
    return _resnet(Bottleneck, [3, 4, 6, 3], rngs=rngs, width_per_group=64 * 2, **kwargs)


def wide_resnet101_2(
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:

    """
    return _resnet(Bottleneck, [3, 4, 23, 3], rngs=rngs, width_per_group=64 * 2, **kwargs)
