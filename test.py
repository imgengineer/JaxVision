import jax
from flax import nnx
from models.shufflenetv2 import shufflenet_v2_x0_5

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1, 224, 224, 3))
model = shufflenet_v2_x0_5(
    rngs=nnx.Rngs(0),
    num_classes=10,
)
print(model(x).shape)
