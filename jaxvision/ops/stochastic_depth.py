import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from


@dataclasses.dataclass(repr=False)
class StochasticDepth(Module):
    rate: float
    broadcast_dims: Sequence[int] = ()
    deterministic: bool = False
    mode: str = "row"
    rng_collection: str = "dropout"
    rngs: rnglib.Rngs | None = None

    def __call__(
        self,
        inputs: jax.Array,
        *,
        deterministic: bool | None = None,
        rngs: rnglib.Rngs | None = None,
    ) -> jax.Array:
        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to Dropout
          as either a __call__ argument or class attribute""",
        )

        if (self.rate == 0.0) or deterministic:
            return inputs

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            return jnp.zeros_like(inputs)

        rngs = first_from(
            rngs,
            self.rngs,
            error_msg="""`deterministic` is False, but no `rngs` argument was provided to Dropout
        as either a __call__ argument or class attribute.""",
        )
        keep_prob = 1.0 - self.rate
        rng = rngs[self.rng_collection]()
        # Zeroes randomly selected rows from the batch.
        # Mask shape: (batch_size, 1, 1, ...) - broadcasting will handle the rest
        # Randomly zeroes the entire input.
        # Mask shape: (1, 1, 1, ...) - will broadcast to input.shape
        broadcast_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1) if self.mode == "row" else (1,) * inputs.ndim

        mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, inputs.shape)
        return jax.lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
