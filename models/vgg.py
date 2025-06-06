from functools import partial
from typing import Any, cast

import jax
from flax import nnx
from jax import Array


class VGG(nnx.Module):
    def __init__(
        self,
        features: nnx.Module,
        num_classes: int = 1000,
        dropout: float = 0.5,
        *,
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

    def __call__(self, x: Array) -> Array:
        x = self.features(x)
        batch_size, _, _, channels = x.shape
        x = jax.image.resize(x, (batch_size, 7, 7, channels), method=jax.image.ResizeMethod.LINEAR)
        x = x.reshape(batch_size, -1)
        return self.classifier(x)


def make_layers(
    cfg: list[str | int],
    batch_norm: bool = False,  # noqa: FBT001, FBT002
    *,
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


def _vgg(cfg: str, batch_norm: bool, *, rngs: nnx.Rngs, **kwargs):  # noqa: FBT001
    features = make_layers(cfgs[cfg], batch_norm=batch_norm, rngs=rngs)
    return VGG(features=features, rngs=rngs, **kwargs)


# --- VGG 模型工厂函数 (使用 partial 优化) ---
# 修正: 确保 rngs 能够正确传递到 _vgg 函数
def vgg11(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("A", False, rngs=rngs, **kwargs)  # noqa: FBT003


def vgg11_bn(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("A", True, rngs=rngs, **kwargs)  # noqa: FBT003


def vgg13(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("B", False, rngs=rngs, **kwargs)  # noqa: FBT003


def vgg13_bn(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("B", True, rngs=rngs, **kwargs)  # noqa: FBT003


def vgg16(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("D", False, rngs=rngs, **kwargs)  # noqa: FBT003


def vgg16_bn(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("D", True, rngs=rngs, **kwargs)  # noqa: FBT003


def vgg19(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("E", False, rngs=rngs, **kwargs)  # noqa: FBT003


def vgg19_bn(rngs: nnx.Rngs, **kwargs: Any) -> VGG:
    return _vgg("E", True, rngs=rngs, **kwargs)  # noqa: FBT003
