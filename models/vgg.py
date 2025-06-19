from functools import partial
from typing import Any, cast

import jax
from flax import nnx
from jax import Array

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


class VGG(nnx.Module):
    def __init__(
        self,
        features: nnx.Module,
        num_classes: int = 1000,
        dropout: float = 0.5,
        *,
        init_weights: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.features = features
        self.classifier = nnx.Sequential(
            nnx.Linear(512 * 7 * 7, 4096, rngs=rngs),
            nnx.relu,
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(4096, 4096, rngs=rngs),
            nnx.relu,
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(4096, num_classes, rngs=rngs),
        )
        if init_weights:
            for _, m in self.iter_modules():
                if isinstance(m, nnx.Conv):
                    m.kernel_init = nnx.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")
                    if m.bias is not None:
                        m.bias_init = nnx.initializers.constant(0)
                    elif isinstance(m, nnx.BatchNorm):
                        m.scale_init = nnx.initializers.constant(1)
                        m.bias_init = nnx.initializers.constant(0)
                    elif isinstance(m, nnx.Linear):
                        m.kernel_init = nnx.initializers.normal(stddev=0.01)
                        m.bias_init = nnx.initializers.constant(0)

    def __call__(self, x: Array) -> Array:
        x = self.features(x)
        batch_size, _, _, channels = x.shape
        x = jax.image.resize(x, (batch_size, 7, 7, channels), method=jax.image.ResizeMethod.LINEAR)
        x = x.reshape(batch_size, -1)
        return self.classifier(x)


def make_layers(
    cfg: list[str | int],
    *,
    batch_norm: bool = False,
    rngs: nnx.Rngs,
) -> nnx.Sequential:
    layers: list[nnx.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2), padding="SAME")]
        else:
            v = cast("int", v)
            conv2d = nnx.Conv(in_channels, v, kernel_size=(3, 3), padding="SAME", rngs=rngs)
            if batch_norm:
                layers += [conv2d, nnx.BatchNorm(v, rngs=rngs), nnx.relu]
            else:
                layers += [conv2d, nnx.relu]
            in_channels = v
    return nnx.Sequential(*layers)


cfgs: dict[str, list[str | int]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(cfg: str, *, batch_norm: bool, rngs: nnx.Rngs, **kwargs):
    features = make_layers(cfgs[cfg], batch_norm=batch_norm, rngs=rngs)
    return VGG(features=features, rngs=rngs, **kwargs)


# --- VGG 模型工厂函数 (使用 partial 优化) ---
# 修正: 确保 rngs 能够正确传递到 _vgg 函数
def vgg11(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("A", batch_norm=False, rngs=rngs, **kwargs)


def vgg11_bn(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("A", batch_norm=True, rngs=rngs, **kwargs)


def vgg13(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("B", batch_norm=False, rngs=rngs, **kwargs)


def vgg13_bn(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("B", batch_norm=True, rngs=rngs, **kwargs)


def vgg16(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("D", batch_norm=False, rngs=rngs, **kwargs)


def vgg16_bn(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("D", batch_norm=True, rngs=rngs, **kwargs)


def vgg19(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("E", batch_norm=False, rngs=rngs, **kwargs)


def vgg19_bn(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("E", batch_norm=True, rngs=rngs, **kwargs)
