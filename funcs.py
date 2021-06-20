# Import packages
import numpy as np
import plotly.graph_objects as go
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax.ops
from funcs import *

# Function to split parameters into matrices and vectors
def par_split(par, n_hidden_ns):
  n_hidden_ns = 10
  splits = n_hidden_ns * (np.arange(3) + 1)
  W0 = par[:splits[0]].reshape(1, n_hidden_ns)
  b0 = par[splits[0]:splits[1]].reshape(1, n_hidden_ns)
  W1 = par[splits[1]:splits[2]].reshape(n_hidden_ns, 1)
  b1 = par[splits[2]:].reshape(1, 1)
  return(W0, b0, W1, b1)

# Function to update RNG key and initialise weights and biases
def init_par(x, y, key, n_par, n_hidden_ns):
  key, subkey = jax.random.split(key)
  par = random.normal(subkey, (n_par, ))

  # Plot initial neuron outputs
  plot_neurons(par, x, y, n_hidden_ns)

  # Return parameters and new RNG key
  return par, key

# Loss function
def loss(x, par, n_hidden_ns, y):
  W0, b0, W1, b1 = par_split(par, n_hidden_ns)
  N0 = jnp.maximum(x @ W0 + b0, 0)
  N1 = N0 @ W1 + b1
  return ((y - N1)**2).reshape(())

# Vectorised loss function
vloss = vmap(loss, in_axes = (0, None, None, 0))

# Compiled vectorised loss function
vloss_jit = jit(vloss)

# Gradient function
grad_fn = grad(loss, argnums=(1))

# Vectorised gradient function
vgrad = vmap(grad_fn, in_axes = (0, None, None, 0))

# Compiled vectorised gradient function
vgrad_jit = jit(vgrad)

# Function to combine lines for neuron output animation
def make_plot_data(par, x, y, n_hidden_ns):
  # Find neuron outputs
  W0, b0, W1, b1 = par_split(par, n_hidden_ns)
  N0 = jnp.maximum(x @ W0 + b0, 0)
  N1 = N0 @ W1 + b1

  # Make plot for target function
  data = [go.Scatter(x=x[:, 0], y=y[:, 0], name="y")]

  # Make plots for neurons in hidden layer
  # for n in np.arange(n_hidden_ns):
  #   data = data + [go.Scatter(
  #       x=x[:, 0], y=N0[:, n], line_color='grey', 
  #       name=f"N0{n} = {W0[0, n]:.2f} * x + {b0[0, n]:.2f}")]

  # Make plot for network output
  data = data + [go.Scatter(x=x[:, 0], y=N1[:, 0], line_color='red', name=f"N10")]

  return data

# Function to plot neuron outputs
def plot_neurons(par, x, y, n_hidden_ns):
  # Find neuron outputs
  W0, b0, W1, b1 = par_split(par, n_hidden_ns)
  N0 = jnp.maximum(x @ W0 + b0, 0)
  N1 = N0 @ W1 + b1

  # Make plot for target function
  data = [go.Scatter(x=x[:, 0], y=y[:, 0], name="y")]

  # Make plots for neurons in hidden layer
  for n in np.arange(n_hidden_ns):
    data = data + [go.Scatter(
        x=x[:, 0], y=N0[:, n], line_color='grey', 
        name=f"N0{n} = {W0[0, n]:.2f} * x + {b0[0, n]:.2f}")]

  # Make plot for network output
  data = data + [go.Scatter(x=x[:, 0], y=N1[:, 0], line_color='red', name=f"N10")]  

  # Setup layout
  layout = dict(
      title='Two-layer neuron outputs', xaxis_title="x", 
      yaxis_title='outputs', autosize=False, width=600, height=400
  )

  # Make plot
  fig = go.Figure(data=data, layout=layout)
  fig.update_traces(hoverinfo='skip')
  fig.show()

# Function to make frame for animation
def make_frame(par, x, y, n_hidden_ns):
  # Make plot data
  data = make_plot_data(par, x, y, n_hidden_ns)
                            
  # Setup layout
  layout = dict(
      title='Two-layer neuron outputs', xaxis_title="x", 
      yaxis_title='outputs', autosize=False, width=600, height=400
  )

  return go.Frame(data=data, layout=layout)

# Function to plot animation of neuron outputs
def plot_animation(par, x, y, n_hidden_ns, frame_list):
  # Buttons for animation
  play_but = dict(label="Play", method="animate", 
                  args=[None, {"transition": {"duration": 0},
                               "frame": {"duration": 500}}])
  pause_but = dict(label="Pause", method="animate",
                  args=[None, {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", 
                                "transition": {"duration": 0}}]) 
  
  # Make animation
  fig = go.Figure(
    data = make_plot_data(par, x, y, n_hidden_ns), 
    layout = go.Layout(
        autosize=False, width=600, height=400, xaxis_title="x", 
        title='Learning animation', yaxis_title='outputs', 
        updatemenus=[dict(type="buttons", buttons=[play_but, pause_but])]
    ),
    frames = frame_list
  )
  fig.update_traces(hoverinfo='skip')
  fig.show()

# Function to plot loss
def plot_loss(loss_vec):
  layout = dict(
      title='Loss over gradient descent', xaxis_title="Iteration", 
      yaxis_title='Loss', autosize=False, width=600, height=400
  )
  fig = go.Figure(data=go.Scatter(y=loss_vec), layout=layout)
  fig.update_traces(hoverinfo='skip')
  fig.show()

# Gradient descent function
def grad_desc(par, x, y, n_hidden_ns, key, lr = 1e-5, tol = 1e-6, max_it = int(5e2)):
  # Loss, iteration counter, frames for animation
  loss_vec = jnp.zeros(max_it)
  loss_vec = loss_vec.at[0].set(jnp.sum(vloss_jit(x, par, n_hidden_ns, y)))
  # loss_vec = loss_vec.at[0].set(jnp.sum(vloss(x, par, n_hidden_ns, y)))
  count = 0
  tot_reinit = 0
  frame_list = []
  
  # Loop until change in loss smaller than tolerance or iteration limit reached
  while True:
    # Make an animation frame at regular intervals
    if (count % 1 == 0):
      frame_list = frame_list + [make_frame(par, x, y, n_hidden_ns)]

    key, subkey = jax.random.split(key)
    n_samp = 101
    idx = random.choice(subkey, np.arange(101), shape=((n_samp, )), replace=False)
    x_samp = x[idx, 0].reshape(n_samp, 1)
    y_samp = y[idx, 0].reshape(n_samp, 1)

    # Find gradient over whole dataset
    par_g = jnp.sum(vgrad_jit(x_samp, par, n_hidden_ns, y_samp), axis = 0)
    # par_g = jnp.sum(vgrad(x_samp, par, n_hidden_ns, y_samp), axis = 0)
    
    # Step down gradient
    par = par - lr * par_g

    # If gradient of loss w.r.t. parameter is zero reinitialise it
    # if (jnp.sum(par_g == 0) > 0):
    #   n_reinit = jnp.sum(par_g == 0)
    #   new_par, key = init_par(key, n_reinit)
    #   par = par.at[jnp.where(par_g == 0)].set(new_par)
    #   tot_reinit = tot_reinit + n_reinit

    # Update loss and iteration counter
    count = count + 1
    loss_vec = loss_vec.at[count].set(jnp.sum(vloss_jit(x_samp, par, n_hidden_ns, y_samp)))
    # loss_vec = loss_vec.at[count].set(jnp.sum(vloss(x_samp, par, n_hidden_ns, y_samp)))

    # If stopping condition met print it and stop
    # if (loss_vec[count - 1] - loss_vec[count] < tol):
    #   print(f"Loss change smaller than tolerance on iteration {count}")
    #   print(f"Loss change {loss_vec[count] - loss_vec[count - 1]}")
    #   break
    if (count + 1 == max_it):
      print(f"Reached maximum number of iterations")
      break

  # Print number of parameters reinitialised due to zero gradients
  print(f"Params reinitialised due to zero gradients {tot_reinit} times")
  print(f"Final number of hidden neurons {n_hidden_ns}")

  # Plot loss
  plot_loss(loss_vec)
  print(loss_vec[::10])

  # Plot learning animation
  plot_animation(par, x, y, n_hidden_ns, frame_list)

  # Return updated parameters
  return(par, n_hidden_ns, key)

# Compile gradient descent function - doesn't work
# grad_desc_jit = jit(grad_desc)