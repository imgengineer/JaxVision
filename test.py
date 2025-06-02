import jax
from flax import nnx
from models.vision_transformer import vit_b_16

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1, 224, 224, 3))
model = vit_b_16(
    rngs=nnx.Rngs(0),
    num_classes=10,
)
print(model(x).shape)
