# Import packages
import plotly.graph_objects as go
import jax.numpy as jnp
from toynnjax.tools import *

# Function to combine lines for neuron output animation
def make_plot_data(par, x, y):
    # Find neuron outputs
    W0, b0, W1, b1 = par_split(par)
    N0 = jnp.maximum(x @ W0 + b0, 0)
    N1 = N0 @ W1 + b1

    # Make plot for target function
    data = [go.Scatter(x=x[:, 0], y=y[:, 0], name="y")]

    # Make plots for neurons in hidden layer
    for n in jnp.arange(b0.shape[1]):
        data = data + [go.Scatter(
            x=x[:, 0], y=N0[:, n], line_color='grey', 
            name=f"N0{n} = {W0[0, n]:.2f} * x + {b0[0, n]:.2f}")]

    # Make plot for network output
    data = data + [go.Scatter(x=x[:, 0], y=N1[:, 0], line_color='red', name=f"N10")]

    return data

# Function to plot neuron outputs
def plot_neurons(par, x, y):
    # Find neuron outputs
    W0, b0, W1, b1 = par_split(par)
    N0 = jnp.maximum(x @ W0 + b0, 0)
    N1 = N0 @ W1 + b1

    # Make plot for target function
    data = [go.Scatter(x=x[:, 0], y=y[:, 0], name="y")]

    # Make plots for neurons in hidden layer
    for n in jnp.arange(b0.shape[1]):
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
def make_frame(par, x, y):
    # Make plot data
    data = make_plot_data(par, x, y)

    # Setup layout
    layout = dict(
      title='Two-layer neuron outputs', xaxis_title="x", 
      yaxis_title='outputs', autosize=False, width=600, height=400
    )

    return go.Frame(data=data, layout=layout)

# Function to plot animation of neuron outputs
def plot_animation(par, x, y, frame_list):
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
    data = make_plot_data(par, x, y), 
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