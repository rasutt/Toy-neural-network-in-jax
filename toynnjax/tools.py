# Import packages
import jax.numpy as jnp
from jax import random

# Function to split parameter vector into weight matrices and bias vectors
def par_split(par):
#     has to be this explicit to get jit to work
    n_h = 10
    W0 = par[:n_h].reshape(1, n_h)
    b0 = par[n_h:(2 * n_h)].reshape(1, n_h)
    W1 = par[(2 * n_h):(3 * n_h)].reshape(n_h, 1)
    b1 = par[(3 * n_h):].reshape(1, 1)
    return(W0, b0, W1, b1)

# Function to update RNG key and initialise weights and biases
def init_par(x, y, key, n_par):
    key, subkey = random.split(key)
    par = random.uniform(subkey, (n_par, )) * 2 - 1
    
#     Pytorch default (Kaimling) U(-sqrt(k), sqrt(k)), where k = number of inputs to layer
    n_h = (n_par - 1) // 3
    lay_2_idx = 2 * n_h
    par = par.at[lay_2_idx:].set(par[lay_2_idx:] / jnp.sqrt(n_h)) 

#     Return parameters and new RNG key
    return par, key

