from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.image import ResizeMethod


class GoogLeNet(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,  # noqa: FBT001, FBT002
        transform_input: bool = False,  # noqa: FBT001, FBT002
        blocks: list[Callable[..., nnx.Module]] | None = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        *,
        rngs: nnx.Rngs,
        training: bool = True,
    ):
        super().__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if len(blocks) != 3:  # noqa: PLR2004
            msg = f"blocks length should be 3 instead of {len(blocks)}"
            raise ValueError(msg)

        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.training = training

        # 基础卷积层
        self.conv1 = conv_block(3, 64, kernel_size=(7, 7), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv2 = conv_block(64, 64, kernel_size=(1, 1), rngs=rngs)
        self.conv3 = conv_block(64, 192, kernel_size=(3, 3), padding="SAME", rngs=rngs)

        # Inception 3层
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32, rngs=rngs)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64, rngs=rngs)

        # Inception 4层
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64, rngs=rngs)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64, rngs=rngs)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64, rngs=rngs)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64, rngs=rngs)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128, rngs=rngs)

        # Inception 5层
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128, rngs=rngs)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128, rngs=rngs)

        # 辅助分类器
        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux, rngs=rngs)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux, rngs=rngs)
        else:
            self.aux1 = None
            self.aux2 = None

        # 最终分类层
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.fc = nnx.Linear(1024, num_classes, rngs=rngs)

        # 预定义的池化层（不需要参数）  # noqa: RUF003
        self.maxpool1 = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        self.maxpool2 = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        self.maxpool3 = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        self.maxpool4 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2), padding="SAME")

    def _transform_input(self, x: Array) -> Array:
        if self.transform_input:
            # ImageNet normalization
            mean = jnp.array([0.485, 0.456, 0.406])
            std = jnp.array([0.229, 0.224, 0.225])
            x = (x - mean) / std
        return x

    def _forward(self, x: Array) -> tuple[Array, Array | None, Array | None]:
        # Initial layers: N x 224 x 224 x 3 -> N x 28 x 28 x 192
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # Inception 3 blocks: N x 28 x 28 x 192 -> N x 14 x 14 x 480
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4 blocks with auxiliary classifiers
        x = self.inception4a(x)  # N x 14 x 14 x 512

        aux1 = None
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)  # N x 14 x 14 x 528

        aux2 = None
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)  # N x 14 x 14 x 832
        x = self.maxpool4(x)  # N x 7 x 7 x 832

        # Inception 5 blocks and final classification
        x = self.inception5a(x)
        x = self.inception5b(x)  # N x 7 x 7 x 1024

        # Global average pooling and classification
        x = jnp.mean(x, axis=(1, 2))  # N x 1024
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux2, aux1

    def __call__(self, x: Array):
        x = self._transform_input(x)
        x, aux2, aux1 = self._forward(x)

        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x


class Inception(nnx.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Callable[..., nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        # Branch 1: 1x1 conv
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=(1, 1), rngs=rngs)

        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2 = nnx.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=(1, 1), rngs=rngs),
            conv_block(ch3x3red, ch3x3, kernel_size=(3, 3), padding="SAME", rngs=rngs),
        )

        # Branch 3: 1x1 conv -> 3x3 conv (originally 5x5)
        self.branch3 = nnx.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=(1, 1), rngs=rngs),
            conv_block(ch5x5red, ch5x5, kernel_size=(3, 3), padding="SAME", rngs=rngs),
        )

        # Branch 4: maxpool -> 1x1 conv
        self.branch4 = nnx.Sequential(
            partial(nnx.max_pool, window_shape=(3, 3), strides=(1, 1), padding="SAME"),
            conv_block(in_channels, pool_proj, kernel_size=(1, 1), rngs=rngs),
        )

    def __call__(self, x: Array) -> Array:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return jnp.concat([branch1, branch2, branch3, branch4], axis=3)


class InceptionAux(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Callable[..., nnx.Module] | None = None,
        dropout: float = 0.7,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.conv = conv_block(in_channels, 128, kernel_size=(1, 1), rngs=rngs)
        self.fc1 = nnx.Linear(2048, 1024, rngs=rngs)
        self.fc2 = nnx.Linear(1024, num_classes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        # Adaptive average pooling to 4x4
        x = jax.image.resize(x, shape=(x.shape[0], 4, 4, x.shape[-1]), method=ResizeMethod.LINEAR)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class BasicConv2d(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs, **kwargs):
        super().__init__()
        self.conv = nnx.Conv(in_channels, out_channels, use_bias=False, rngs=rngs, **kwargs)
        self.bn = nnx.BatchNorm(out_channels, epsilon=0.001, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = self.conv(x)
        x = self.bn(x)
        return nnx.relu(x)
