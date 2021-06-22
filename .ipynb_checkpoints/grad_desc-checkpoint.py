# Import packages
import numpy as np
import plotly.graph_objects as go
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax.ops
from funcs import *

# Gradient descent function
def grad_desc(par, x, y, n_hidden_ns, key, lr = 1e-5, max_it = int(5e2)):
    # Loss, iteration counter, frames for animation
    loss_vec = jnp.zeros(max_it)
    loss_vec = loss_vec.at[0].set(jnp.sum(vloss_jit(x, par, n_hidden_ns, y)))
    count = 0
    tot_reinit = 0
    frame_list = []

    # Loop until change in loss smaller than tolerance or iteration limit reached
    while True:
        # Make an animation frame at regular intervals
        if (count % 20 == 0):
            frame_list = frame_list + [make_frame(par, x, y, n_hidden_ns)]

        key, subkey = jax.random.split(key)
        n_samp = 101
        idx = random.choice(subkey, np.arange(101), shape=((n_samp, )), replace=False)
        x_samp = x[idx, 0].reshape(n_samp, 1)
        y_samp = y[idx, 0].reshape(n_samp, 1)

        # Find gradient over whole dataset
        par_g = jnp.sum(vgrad_jit(x_samp, par, n_hidden_ns, y_samp), axis = 0)

        # Step down gradient
        par = par - lr * par_g

        # Update loss and iteration counter
        count = count + 1
        loss_vec = loss_vec.at[count].set(jnp.sum(vloss_jit(x_samp, par, n_hidden_ns, y_samp)))

        if (count + 1 == max_it):
            print(f"Reached maximum number of iterations")
            break

    # Plot loss and learning animation
    plot_loss(loss_vec)
    plot_animation(par, x, y, n_hidden_ns, frame_list)

    # Return updated parameters
    return(par, key)