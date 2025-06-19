import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from ops.misc import MLP, Conv2dNormActivation

__all__ = [
    "VisionTransformer",
    "vit_b_16",
    "vit_b_32",
    "vit_h_14",
    "vit_l_16",
    "vit_l_32",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nnx.Module] = nnx.BatchNorm
    activation_layer: Callable[..., nnx.Module] = nnx.relu


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float, *, rngs: nnx.Rngs):
        super().__init__(
            in_dim,
            [mlp_dim, in_dim],
            activation_layer=nnx.gelu,
            dropout=dropout,
            rngs=rngs,
        )
        for _, m in self.iter_modules():
            if isinstance(m, nnx.Linear):
                m.kernel_init = nnx.initializers.xavier_uniform()
                if m.bias is not None:
                    m.bias_init = nnx.initializers.normal(stddev=1e-6)


class EncoderBlock(nnx.Module):
    """Transformer encoder block"""

    def __init__(  # noqa: PLR0913
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nnx.Module] = partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim, rngs=rngs)
        self.self_attention = nnx.MultiHeadAttention(
            num_heads,
            hidden_dim,
            dropout_rate=attention_dropout,
            decode=False,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim, rngs=rngs)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout, rngs=rngs)

    def __call__(self, inputs: Array) -> Array:
        x = self.ln_1(inputs)
        x = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + inputs

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nnx.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(  # noqa: PLR0913
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nnx.Module] = partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.pos_embedding = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(key=rngs.params(), shape=(1, seq_length, hidden_dim))
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        layers: list[nnx.Module] = []
        for _ in range(num_layers):
            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                    rngs=rngs,
                )
            )
        self.layers = nnx.Sequential(*layers)
        self.ln = norm_layer(hidden_dim, rngs=rngs)

    def __call__(self, inputs: Array) -> Array:
        inputs = inputs + self.pos_embedding.value
        return self.ln(self.layers(self.dropout(inputs)))


class VisionTransformer(nnx.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(  # noqa: PLR0913
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: int | None = None,
        norm_layer: Callable[..., nnx.Module] = partial(nnx.LayerNorm, epsilon=1e-6),  # noqa: B008
        conv_stem_configs: list[ConvStemConfig] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2104.14881
            self.seq_proj_layers = {}
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                layer_name = f"conv_bn_relu_{i}"
                self.conv_proj[layer_name] = Conv2dNormActivation(
                    in_channels=prev_channels,
                    out_channels=conv_stem_layer_config.out_channels,
                    kernel_size=conv_stem_layer_config.kernel_size,
                    stride=conv_stem_layer_config.stride,
                    norm_layer=conv_stem_layer_config.norm_layer,
                    activation_layer=conv_stem_layer_config.activation_layer,
                    rngs=rngs,
                )
                prev_channels = conv_stem_layer_config.out_channels
            self.seq_proj_layers["conv_last"] = nnx.Conv(
                prev_channels,
                hidden_dim,
                kernel_size=(1, 1),
                rngs=rngs,
            )
            self.conv_proj = None
        else:
            self.conv_proj = nnx.Conv(
                3,
                hidden_dim,
                kernel_size=(patch_size, patch_size),
                strides=(patch_size, patch_size),
                rngs=rngs,
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nnx.Param(nnx.initializers.zeros(rngs.params(), (1, 1, hidden_dim)))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            rngs=rngs,
        )

        self.seq_length = seq_length

        self.heads_layers = {}
        if representation_size is None:
            self.heads_layers["head"] = nnx.Linear(hidden_dim, num_classes, rngs=rngs)
        else:
            self.heads_layers["pre_logits"] = nnx.Linear(hidden_dim, representation_size, rngs=rngs)
            self.heads_layers["act"] = nnx.tanh
            self.heads_layers["head"] = nnx.Linear(representation_size, num_classes, rngs=rngs)

        self.heads = lambda x: self._run_heads(x, self.heads_layers)

        if isinstance(self.conv_proj, nnx.Conv):
            # Init the patchify stem
            fan_in = self.conv_proj.in_features * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            self.conv_proj.kernel_init = nnx.initializers.truncated_normal(stddev=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                self.conv_proj.bias_init = nnx.initializers.zeros_init()
        elif self.seq_proj_layers["conv_last"] is not None and isinstance(self.seq_proj_layers["conv_last"], nnx.Conv):
            self.seq_proj_layers["conv_last"].kernel_init = nnx.initializers.normal(
                stddev=math.sqrt(2.0 / self.seq_proj_layers["conv_last"].out_features)
            )
        if hasattr(self.heads_layers, "pre_logits") and isinstance(self.heads_layers["pre_logits"], nnx.Linear):
            fan_in = self.heads_layers["pre_logits"].in_features
            self.heads_layers["pre_logits"].kernel_init = nnx.initializers.truncated_normal(
                stddev=math.sqrt(1 / fan_in)
            )
            self.heads_layers["pre_logits"].bias_init = nnx.initializers.zeros_init()
        if isinstance(self.heads_layers["head"], nnx.Linear):
            self.heads_layers["head"].kernel_init = nnx.initializers.zeros_init()
            self.heads_layers["head"].bias_init = nnx.initializers.zeros_init()

    def _run_heads(self, x, layers_dict):
        if "pre_logits" in layers_dict:
            x = layers_dict["pre_logits"](x)
            x = layers_dict["act"](x)
        x = layers_dict["head"](x)
        return x

    def _process_input(self, x: Array) -> Array:
        n, h, w, c = x.shape
        p = self.patch_size

        n_h = h // p
        n_w = w // p

        # (n, h, w, c) -> (n, n_h, n_w, hidden_dim)
        x = self.conv_proj(x)
        # (n, n_h, n_w, hidden_dim) -> (n, (n_h * n_w), hidden_dim)
        return x.reshape(n, n_h * n_w, self.hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension

    def __call__(self, x: Array) -> Array:
        # Reshape the input Tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = jnp.broadcast_to(self.class_token.value, (n, *self.class_token.value.shape[1:]))
        x = jnp.concat([batch_class_token, x], axis=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return self.heads(x)


def _vision_transformer(  # noqa: PLR0913
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> VisionTransformer:
    image_size = kwargs.pop("image_size", 224)

    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        rngs=rngs,
        **kwargs,
    )


def vit_b_16(*, rngs: nnx.Rngs, **kwargs) -> VisionTransformer:
    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        rngs=rngs,
        **kwargs,
    )


def vit_b_32(*, rngs: nnx.Rngs, **kwargs) -> VisionTransformer:
    return _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        rngs=rngs,
        **kwargs,
    )


def vit_l_16(*, rngs: nnx.Rngs, **kwargs) -> VisionTransformer:
    return _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        rngs=rngs,
        **kwargs,
    )


def vit_l_32(*, rngs: nnx.Rngs, **kwargs) -> VisionTransformer:
    return _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        rngs=rngs,
        **kwargs,
    )


def vit_h_14(*, rngs: nnx.Rngs, **kwargs) -> VisionTransformer:
    return _vision_transformer(
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        rngs=rngs,
        **kwargs,
    )
