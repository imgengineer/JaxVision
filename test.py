import jax
from flax import nnx

from models.vgg import vgg11

rngs = nnx.Rngs(0)
x = jax.random.normal(rngs.params(), (1, 224, 224, 3))
model = vgg11(
    rngs=rngs,
    num_classes=10,
)
print(model(x).shape)
