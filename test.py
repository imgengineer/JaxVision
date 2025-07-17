import jax
from flax import nnx

from jaxvision.models.swin_transformer import swin_v2_t

rngs = nnx.Rngs(0)
x = jax.random.normal(jax.random.key(0), (1, 224, 224, 3))
model = swin_v2_t(
    rngs=rngs,
    num_classes=10,
)
model.eval()

print(model(x).shape)
