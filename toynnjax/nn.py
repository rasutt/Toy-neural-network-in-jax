# Import packages
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from toynnjax.tools import *
from toynnjax.plot import *

# Loss function
def loss(x, par, y):
    W0, b0, W1, b1 = par_split(par)
    N0 = jnp.maximum(x @ W0 + b0, 0)
    N1 = N0 @ W1 + b1
    return ((y - N1)**2).reshape(())

# Compiled vectorised loss and gradient functions
vloss_jit = jit(vmap(loss, in_axes = (0, None, 0)))
vgrad_jit = jit(vmap(grad(loss, argnums=(1)), in_axes = (0, None, 0)))

# Gradient descent function
def grad_desc(par, x, y, key, lr = 1e-5, max_it = int(5e2)):
    # Loss, iteration counter, frames for animation
    loss_vec = jnp.zeros(max_it)
    loss_vec = loss_vec.at[0].set(jnp.sum(vloss_jit(x, par, y)))
    count = 0
    tot_reinit = 0
    frame_list = []

    # Loop until change in loss smaller than tolerance or iteration limit reached
    while True:
        # Make an animation frame at regular intervals
        if (count % 20 == 0):
            frame_list = frame_list + [make_frame(par, x, y)]

        key, subkey = random.split(key)
        n_samp = 101
        idx = random.choice(subkey, jnp.arange(101), shape=((n_samp, )), replace=False)
        x_samp = x[idx, 0].reshape(n_samp, 1)
        y_samp = y[idx, 0].reshape(n_samp, 1)

        # Find gradient over whole dataset
        par_g = jnp.sum(vgrad_jit(x_samp, par, y_samp), axis = 0)

        # Step down gradient
        par = par - lr * par_g

        # Update loss and iteration counter
        count = count + 1
        loss_vec = loss_vec.at[count].set(jnp.sum(vloss_jit(x_samp, par, y_samp)))

        if (count + 1 == max_it):
            print(f"Reached maximum number of iterations")
            break

    # Plot loss and learning animation
    plot_loss(loss_vec)
    plot_animation(par, x, y, frame_list)

    # Return updated parameters
    return(par, key)