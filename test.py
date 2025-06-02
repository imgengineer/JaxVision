import jax
from flax import nnx
from models.vision_transformer import vit_b_16
import jax.numpy as jnp

rngs = nnx.Rngs(0)
x = jax.random.normal(rngs.params(), (1, 224, 224, 3))
print(x.transpose(1, 2, 3, 0).shape)
# model = vit_b_16(
#     rngs=rngs,
#     num_classes=10,
# )
# print(model(x).shape)
