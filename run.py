from jax import jit, grad, vmap
from jax import random
import jax.numpy as jnp
import numpy as np
import time


# Define a function
def poly(x):
    sum = 0.0
    n = 4
    for i in range(n):
        sum += x**i
    return sum


# Generate random input
NUM_SAMPLES = 1000000
x_np = np.random.rand(NUM_SAMPLES).astype(np.float32)
x_jnp = jnp.array(x_np)


# Numpy
start = time.time()
poly(x_np)
np_time = time.time() - start

# JAX
start = time.time()
jit(poly)(x_jnp).block_until_ready()
jax_time = time.time() - start

# Print results
jax_times_faster = np_time / jax_time
if jax_times_faster > 1.0:
    print(f"JAX is {jax_times_faster:.2f}x faster than Numpy")
else:
    print(f"Numpy is {1.0 / jax_times_faster:.2f}x faster than JAX")
