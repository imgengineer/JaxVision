import jax
from flax import nnx

from models.googlenet import googlenet

rngs = nnx.Rngs(0)
x = jax.random.normal(rngs.params(), (1, 224, 224, 3))
model = googlenet(
    rngs=rngs,
    num_classes=10,
)
model.eval()

y= model(x)
print(y.shape)
