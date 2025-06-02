import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


def stochastic_depth(  # noqa: D417
    rngs: nnx.Rngs,
    input: Array,  # noqa: A002
    p: float,
    mode: str,
    training: bool = True,  # noqa: FBT001, FBT002
) -> Array:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.

    """
    if p < 0.0 or p > 1.0:
        msg = f"drop probability has to between 0 and 1, but got {p}"
        raise ValueError(msg)
    if mode not in ["batch", "row"]:
        msg = f"mode has to be either 'batch' or 'row', but got {mode}"
        raise ValueError(msg)
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":  # noqa: SIM108
        # Zeroes randomly selected rows from the batch.
        # Mask shape: (batch_size, 1, 1, ...) - broadcasting will handle the rest
        size = (input.shape[0],) + (1,) * (input.ndim - 1)
    else:
        # Randomly zeroes the entire input.
        # Mask shape: (1, 1, 1, ...) - will broadcast to input.shape
        size = (1,) * input.ndim
    # Generate a Bernoulli mask.
    # jax.random.bernoulli(key, p) generates True with probability p.
    # Here, we want to 'keep' with probability `survival_rate`.
    # The mask will be True for kept elements, False for dropped elements.
    noise_mask = jax.random.bernoulli(rngs.params(), p=survival_rate, shape=size)

    if survival_rate > 0.0:
        # Convert boolean mask to float (True=1.0, False=0.0) and scale
        noise = noise_mask.astype(input.dtype) / survival_rate
    else:
        # If survival_rate is 0, it means p=1.0 (all dropped).
        # In this case, noise should be all zeros.
        noise = jnp.zeros(size, dtype=input.dtype)
    # Apply the noise to the input
    return input * noise


class StochasticDepth(nnx.Module):
    def __init__(
        self,
        p: float,
        mode: str,
        rngs: nnx.Rngs,
        training: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        super().__init__()
        self.p = p
        self.mode = mode
        self.rngs = rngs
        self.training = training

    def __call__(self, x: Array) -> Array:
        return stochastic_depth(self.rngs, x, self.p, self.mode, self.training)
