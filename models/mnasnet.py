from typing import Any

from flax import nnx
from jax import Array

__all__ = [
    "MNASNet",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
]

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        expansion_factor: int,
        bn_momentum: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if stride not in [1, 2]:
            msg = f"stride should be 1 or 2 instead of {stride}"
            raise ValueError(msg)
        if kernel_size not in [3, 5]:
            msg = f"kernel_size should be 3 or 5 instead of {kernel_size}"
            raise ValueError(msg)
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nnx.Sequential(
            # Pointwise
            nnx.Conv(in_ch, mid_ch, kernel_size=(1, 1), use_bias=False, rngs=rngs),
            nnx.BatchNorm(mid_ch, momentum=bn_momentum, rngs=rngs),
            nnx.relu,
            # Depthwise
            nnx.Conv(
                mid_ch,
                mid_ch,
                kernel_size=(kernel_size, kernel_size),
                padding=(kernel_size // 2, kernel_size // 2),
                strides=(stride, stride),
                feature_group_count=mid_ch,
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(mid_ch, momentum=bn_momentum, rngs=rngs),
            nnx.relu,
            # Linear pointwise. Note that there's no activation.
            nnx.Conv(mid_ch, out_ch, kernel_size=(1, 1), use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_ch, momentum=bn_momentum, rngs=rngs),
        )

    def __call__(self, x: Array) -> Array:
        if self.apply_residual:
            return self.layers(x) + x
        return self.layers(x)


def _stack(  # noqa: PLR0913
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int,
    exp_factor: int,
    repeats: int,
    bn_momentum: float,
    *,
    rngs: nnx.Rngs,
) -> nnx.Sequential:
    """Creates a stack of inverted residuals."""
    if repeats < 1:
        msg = f"repeats should be >= 1, instead got {repeats}"
        raise ValueError(msg)
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum=bn_momentum, rngs=rngs)
    remaining = []
    remaining = [
        _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum=bn_momentum, rngs=rngs)
        for _ in range(1, repeats)
    ]
    return nnx.Sequential(first, *remaining)


def _round_to_multiple_of(val: float, divisor: int, round_up_bias: float = 0.9) -> int:
    """Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88.
    """
    if not 0.0 < round_up_bias < 1.0:
        msg = f"round_up_bias should be greater than 0.0 and smaller than 1.0 instead of {round_up_bias}"
        raise ValueError(msg)
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha: float) -> list[int]:
    """Scales tensor depths as in reference MobileNet code, prefers rounding up
    rather than down.
    """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(nnx.Module):
    """MNASNet, as described in https://arxiv.org/abs/1807.11626. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1.0, num_classes=1000)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    2
    >>> y.nelement()
    1000.
    """

    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(
        self,
        alpha: float,
        num_classes: int = 1000,
        dropout: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            msg = f"alpha should be greater than 0.0 instead of {alpha}"
            raise ValueError(msg)
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [
            # First layer: regular conv.
            nnx.Conv(3, depths[0], kernel_size=(3, 3), padding="SAME", strides=(2, 2), use_bias=False, rngs=rngs),
            nnx.BatchNorm(depths[0], momentum=_BN_MOMENTUM, rngs=rngs),
            nnx.relu,
            # Depthwise separable, no skip.
            nnx.Conv(
                depths[0],
                depths[0],
                kernel_size=(3, 3),
                padding=(1, 1),
                strides=(1, 1),
                feature_group_count=depths[0],
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(depths[0], momentum=_BN_MOMENTUM, rngs=rngs),
            nnx.relu,
            nnx.Conv(
                depths[0],
                depths[1],
                kernel_size=(1, 1),
                padding="SAME",
                strides=(1, 1),
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(depths[1], momentum=_BN_MOMENTUM, rngs=rngs),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM, rngs=rngs),
            _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM, rngs=rngs),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM, rngs=rngs),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM, rngs=rngs),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM, rngs=rngs),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM, rngs=rngs),
            # Final mapping to classifier input.
            nnx.Conv(depths[7], 1280, kernel_size=(1, 1), padding="SAME", strides=(1, 1), use_bias=False, rngs=rngs),
            nnx.BatchNorm(1280, momentum=_BN_MOMENTUM, rngs=rngs),
            nnx.relu,
        ]
        self.layers = nnx.Sequential(*layers)
        self.classifier = nnx.Sequential(nnx.Dropout(rate=dropout, rngs=rngs), nnx.Linear(1280, num_classes, rngs=rngs))

        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv):
                m.kernel_init = nnx.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")
                if m.bias is not None:
                    m.bias_init = nnx.initializers.zeros_init()
            elif isinstance(m, nnx.BatchNorm):
                m.scale_init = nnx.initializers.ones_init()
                m.bias_init = nnx.initializers.zeros_init()
            elif isinstance(m, nnx.Linear):
                m.kernel_init = nnx.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")
                m.bias_init = nnx.initializers.zeros_init()

    def __call__(self, x: Array) -> Array:
        x = self.layers(x)
        x = x.mean(axis=(1, 2))
        return self.classifier(x)


def _mnasnet(alpha: float, *, rngs: nnx.Rngs, **kwargs: Any) -> MNASNet:
    return MNASNet(alpha, rngs=rngs, **kwargs)



def mnasnet0_5(*, rngs: nnx.Rngs, **kwargs: Any) -> MNASNet:
    """MNASNet with depth multiplier of 0.5 from
    `MnasNet: Platform-Aware Neural Architecture Search for Mobile
    <https://arxiv.org/abs/1807.11626>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MNASNet0_5_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MNASNet0_5_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mnasnet.MNASNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MNASNet0_5_Weights
        :members:

    """
    return _mnasnet(0.5, rngs=rngs, **kwargs)


def mnasnet0_75(*, rngs: nnx.Rngs, **kwargs: Any) -> MNASNet:
    """MNASNet with depth multiplier of 0.75 from
    `MnasNet: Platform-Aware Neural Architecture Search for Mobile
    <https://arxiv.org/abs/1807.11626>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MNASNet0_75_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MNASNet0_75_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mnasnet.MNASNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MNASNet0_75_Weights
        :members:

    """
    return _mnasnet(0.75, rngs=rngs, **kwargs)


def mnasnet1_0(*, rngs: nnx.Rngs, **kwargs: Any) -> MNASNet:
    """MNASNet with depth multiplier of 1.0 from
    `MnasNet: Platform-Aware Neural Architecture Search for Mobile
    <https://arxiv.org/abs/1807.11626>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MNASNet1_0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MNASNet1_0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mnasnet.MNASNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MNASNet1_0_Weights
        :members:

    """
    return _mnasnet(1.0, rngs=rngs, **kwargs)


def mnasnet1_3(*, rngs: nnx.Rngs, **kwargs: Any) -> MNASNet:
    """MNASNet with depth multiplier of 1.3 from
    `MnasNet: Platform-Aware Neural Architecture Search for Mobile
    <https://arxiv.org/abs/1807.11626>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MNASNet1_3_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MNASNet1_3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mnasnet.MNASNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MNASNet1_3_Weights
        :members:

    """
    return _mnasnet(1.3, rngs=rngs, **kwargs)
