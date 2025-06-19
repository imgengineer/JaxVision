import warnings
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from flax import nnx
from jax import Array

__all__ = ["Inception3", "inception_v3"]


class InceptionOutputs(NamedTuple):
    logits: Array
    aux_logits: Array | None


_InceptionOutputs = InceptionOutputs


class Inception3(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        num_classes: int = 1000,
        inception_blocks: list[Callable[..., nnx.Module]] | None = None,
        dropout: float = 0.5,
        *,
        rngs: nnx.Rngs,
        init_weights: bool | None = None,
        aux_logits: bool = True,
        transform_input: bool = False,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of inception_v3 will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
                stacklevel=2,
            )
            init_weights = True
        if len(inception_blocks) != 7:  # noqa: PLR2004
            msg = f"length of inception_blocks should be 7 instead of {len(inception_blocks)}"
            raise ValueError(msg)
        self.deterministic = deterministic

        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=(3, 3), rngs=rngs)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.maxpool1 = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2))
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=(1, 1), rngs=rngs)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=(3, 3), rngs=rngs)
        self.maxpool2 = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2))
        self.Mixed_5b = inception_a(192, pool_features=32, rngs=rngs)
        self.Mixed_5c = inception_a(256, pool_features=64, rngs=rngs)
        self.Mixed_5d = inception_a(288, pool_features=64, rngs=rngs)
        self.Mixed_6a = inception_b(288, rngs=rngs)
        self.Mixed_6b = inception_c(768, channels_7x7=128, rngs=rngs)
        self.Mixed_6c = inception_c(768, channels_7x7=160, rngs=rngs)
        self.Mixed_6d = inception_c(768, channels_7x7=160, rngs=rngs)
        self.Mixed_6e = inception_c(768, channels_7x7=192, rngs=rngs)
        self.AuxLogits: nnx.Module | None = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes, rngs=rngs)
        self.Mixed_7a = inception_d(768, rngs=rngs)
        self.Mixed_7b = inception_e(1280, rngs=rngs)
        self.Mixed_7c = inception_e(2048, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.fc = nnx.Linear(2048, num_classes, rngs=rngs)

        if init_weights:
            for _, m in self.iter_modules():
                if isinstance(m, nnx.Conv | nnx.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1
                    m.kernel_init = nnx.initializers.truncated_normal(stddev=stddev, lower=-2, upper=2)
                elif isinstance(m, nnx.BatchNorm):
                    m.scale_init = nnx.initializers.constant(1)
                    m.bias_init = nnx.initializers.constant(0)

    def __call__(self, x: Array):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Array | None = None
        if self.AuxLogits is not None and not self.deterministic:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = x.mean(axis=(1, 2))
        # N x 2048
        x = self.dropout(x)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        if not self.deterministic and self.aux_logits:
            return InceptionOutputs(x, aux)
        return x


class InceptionA(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=(1, 1), rngs=rngs)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=(1, 1), rngs=rngs)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=(5, 5), padding="SAME", rngs=rngs)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=(1, 1), rngs=rngs)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=(3, 3), padding="SAME", rngs=rngs)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: Array) -> Array:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = nnx.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return jnp.concat(outputs, axis=-1)


class InceptionB(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        conv_block: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=(1, 1), rngs=rngs)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=(3, 3), padding=(1, 1), rngs=rngs)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)

    def __call__(self, x: Array) -> Array:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        outputs = [branch3x3, branch3x3dbl, branch_pool]

        return jnp.concat(outputs, axis=-1)


class InceptionC(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=(1, 1), rngs=rngs)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=(1, 1), rngs=rngs)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3), rngs=rngs)
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0), rngs=rngs)

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=(1, 1), rngs=rngs)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0), rngs=rngs)
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3), rngs=rngs)
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0), rngs=rngs)
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3), rngs=rngs)

        self.branch_pool = conv_block(in_channels, 192, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: Array) -> Array:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = nnx.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return jnp.concat(outputs, axis=-1)


class InceptionD(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        conv_block: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=(1, 1), rngs=rngs)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=(1, 1), rngs=rngs)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3), rngs=rngs)
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0), rngs=rngs)
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)

    def __call__(self, x: Array) -> Array:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        outputs = [branch3x3, branch7x7x3, branch_pool]

        return jnp.concat(outputs, axis=-1)


class InceptionE(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        conv_block: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=(1, 1), rngs=rngs)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=(1, 1), rngs=rngs)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), rngs=rngs)
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), rngs=rngs)

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=(1, 1), rngs=rngs)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), rngs=rngs)
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), rngs=rngs)

        self.branch_pool = conv_block(in_channels, 192, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: Array) -> Array:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = jnp.concat(branch3x3, axis=-1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = jnp.concat(branch3x3dbl, axis=-1)

        branch_pool = nnx.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return jnp.concat(outputs, axis=-1)


class InceptionAux(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=(1, 1), rngs=rngs)
        self.conv1 = conv_block(128, 768, kernel_size=(5, 5), rngs=rngs)
        self.fc = nnx.Linear(768, num_classes, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        # N x 768 x 17 x 17
        x = nnx.avg_pool(x, window_shape=(5, 5), strides=(3, 3), padding="SAME")
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = x.mean(axis=(1, 2))
        # N x 768
        return self.fc(x)
        # N x 1000


class BasicConv2d(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs, **kwargs) -> None:
        super().__init__()
        self.conv = nnx.Conv(in_channels, out_channels, use_bias=False, rngs=rngs, **kwargs)
        self.bn = nnx.BatchNorm(out_channels, rngs=rngs, epsilon=0.001)

    def __call__(self, x: Array) -> Array:
        x = self.conv(x)
        x = self.bn(x)
        return nnx.relu(x)


def inception_v3(*, rngs: nnx.Rngs, **kwargs) -> Inception3:
    return Inception3(aux_logits=True, rngs=rngs, **kwargs)
