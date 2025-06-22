from functools import partial

import jax.numpy as jnp
from flax import nnx
from jax import Array

__all__ = ["SqueezeNet", "squeezenet1_0", "squeezenet1_1"]


class Fire(nnx.Module):
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.inplanes = inplanes
        self.squeeze = nnx.Conv(inplanes, squeeze_planes, kernel_size=(1, 1), rngs=rngs)
        self.expand1x1 = nnx.Conv(squeeze_planes, expand1x1_planes, kernel_size=(1, 1), rngs=rngs)
        self.expand3x3 = nnx.Conv(squeeze_planes, expand3x3_planes, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = nnx.relu(self.squeeze(x))
        return jnp.concat(
            [nnx.relu(self.expand1x1(x)), nnx.relu(self.expand3x3(x))],
            axis=-1,
        )


class SqueezeNet(nnx.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5, *, rngs: nnx.Rngs) -> None:
        self.num_classes = num_classes
        if version == "1_0":
            self.features = nnx.Sequential(
                nnx.Conv(3, 96, kernel_size=(7, 7), strides=(2, 2), rngs=rngs),
                nnx.relu,
                partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
                Fire(96, 16, 64, 64, rngs=rngs),
                Fire(128, 16, 64, 64, rngs=rngs),
                Fire(128, 32, 128, 128, rngs=rngs),
                partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
                Fire(256, 32, 128, 128, rngs=rngs),
                Fire(256, 48, 192, 192, rngs=rngs),
                Fire(384, 48, 192, 192, rngs=rngs),
                Fire(384, 64, 256, 256, rngs=rngs),
                partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
                Fire(512, 64, 256, 256, rngs=rngs),
            )

        elif version == "1_1":
            self.features = nnx.Sequential(
                nnx.Conv(3, 64, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
                nnx.relu,
                partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
                Fire(64, 16, 64, 64, rngs=rngs),
                Fire(128, 16, 64, 64, rngs=rngs),
                partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
                Fire(128, 32, 128, 128, rngs=rngs),
                Fire(256, 32, 128, 128, rngs=rngs),
                partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
                Fire(256, 48, 192, 192, rngs=rngs),
                Fire(384, 48, 192, 192, rngs=rngs),
                Fire(384, 64, 256, 256, rngs=rngs),
                Fire(512, 64, 256, 256, rngs=rngs),
            )
        else:
            msg = f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected"
            raise ValueError(msg)

        final_conv = nnx.Conv(512, self.num_classes, kernel_size=(1, 1), rngs=rngs)
        self.classifier = nnx.Sequential(
            nnx.Dropout(rate=dropout, rngs=rngs),
            final_conv,
            nnx.relu,
        )

        for _, m in self.iter_modules():
            if isinstance(m, nnx.Conv):
                if m is final_conv:
                    m.kernel_init = nnx.initializers.normal(stddev=0.01)
                else:
                    m.kernel_init = nnx.initializers.kaiming_uniform()
                if m.bias is not None:
                    m.bias_init = nnx.initializers.constant(0)

    def __call__(self, x: Array) -> Array:
        x = self.features(x)
        x = self.classifier(x)
        return x.mean(axis=(1, 2))


def _squeezenet(
    version: str,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> SqueezeNet:
    return SqueezeNet(version, rngs=rngs, **kwargs)


def squeezenet1_0(*, rngs: nnx.Rngs, **kwargs) -> SqueezeNet:
    return _squeezenet("1_0", rngs=rngs, **kwargs)


def squeezenet1_1(*, rngs: nnx.Rngs, **kwargs) -> SqueezeNet:
    return _squeezenet("1_1", rngs=rngs, **kwargs)
