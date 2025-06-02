import jax
from flax import nnx
from models.shufflenetv2 import  shufflenet_v2_x2_0

rngs = nnx.Rngs(0)
x = jax.random.normal(rngs.params(), (1, 224, 224, 3))
model = shufflenet_v2_x2_0(
    rngs=rngs,
    num_classes=10,
)
print(model(x).shape)
