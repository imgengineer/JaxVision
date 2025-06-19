import jax
from flax import nnx

from jaxvision.models.swin_transformer import swin_t

rngs = nnx.Rngs(0)
x = jax.random.normal(rngs.params(), (1, 224, 224, 3))
model = swin_t(
    rngs=rngs,
    num_classes=10,
)
model.eval()

print(model(x).shape)
# for path, m in model.iter_modules():
#     if isinstance(m, nnx.Linear):
#         print(path, m)
