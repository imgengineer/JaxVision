from functools import partial

import jax
from flax import nnx

__all__ = ["AlexNet", "alexnet"]


class AlexNet(nnx.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *, rngs: nnx.Rngs) -> None:
        self.features = nnx.Sequential(
            # First conv block
            nnx.Conv(3, 64, kernel_size=(11, 11), padding="SAME", rngs=rngs),
            nnx.relu,
            partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
            # Second conv block
            nnx.Conv(64, 192, kernel_size=(5, 5), padding="SAME", rngs=rngs),
            nnx.relu,
            partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
            # Third conv block
            nnx.Conv(192, 384, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu,
            # Fourth conv block
            nnx.Conv(384, 256, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu,
            # Fifth conv block
            nnx.Conv(256, 256, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu,
            partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2)),
        )

        self.classifier = nnx.Sequential(
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(256 * 6 * 6, 4096, rngs=rngs),
            nnx.relu,
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(4096, 4096, rngs=rngs),
            nnx.relu,
            nnx.Linear(4096, num_classes, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.features(x)
        batch_size, _, _, channels = x.shape
        x = jax.image.resize(x, (batch_size, 6, 6, channels), method=jax.image.ResizeMethod.LINEAR)
        x = x.reshape(batch_size, -1)
        return self.classifier(x)


def alexnet(*, rngs: nnx.Rngs, **kwargs) -> AlexNet:
    return AlexNet(rngs=rngs, **kwargs)
