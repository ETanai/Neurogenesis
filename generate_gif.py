# %%
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.optim.lr_scheduler import ExponentialLR
import math
import numpy as np
import pandas as pd
from PIL import Image

# %%
Path_folder = Path(
    r"C:\Users\Admin\Documents\GitHub\Neurogenesis\Modelweighs\Parameters_20240612_134314")
paths_files = [file for file in Path_folder.iterdir() if file.is_file()]

data = []

# Load each file and handle any exceptions
for path in paths_files:
    try:
        loaded_data = torch.load(path)
        data.append(loaded_data['model_state_dict'])
        print(f"Successfully loaded: {path}")
    except Exception as e:
        print(f"Failed to load {path}: {e}")
# %%
# Assuming you have already loaded the state_dicts into a list called 'data'
state_dicts = data

# Initialize a dictionary to store the lists of weights and biases for each layer
layer_params = {}

# Get the keys from the first state_dict
first_state_dict = state_dicts[0].keys()

# Initialize lists in the dictionary for each key
for key in first_state_dict:
    layer_params[key] = []

# Populate the dictionary with weights and biases from each state_dict
for state_dict in state_dicts:
    for key in state_dict:
        layer_params[key].append(state_dict[key])

# Now layer_params contains lists of weights and biases for each layer
for layer, params in layer_params.items():
    print(f"{layer}: {len(params)} elements")
# %%


def pad_tensor(tensor, target_shape):
    """
    Pads a tensor with zeros to match the target shape.
    """
    size = tensor.size()
    new = torch.zeros(target_shape)
    if len(size) == 2:
        new[:size[0], :size[1]] = tensor
    elif len(size) == 1:
        new[:size[0]] = tensor
    return new


padded_layer_params = {}
for layer, tensors in layer_params.items():
    max_shape = tensors[-1].size()
    padded_tensors = [pad_tensor(tensor, max_shape) for tensor in tensors]
    padded_layer_params[layer] = padded_tensors

# # Now padded_layer_params contains the padded tensors for each layer
# for layer, tensors in padded_layer_params.items():
#     print(f"{layer}: {len(tensors)} tensors, each of size {tensors[0].shape}")

# # Example to access padded tensors of a specific layer
# example_layer = 'layer_name_here'
# padded_tensors_example = padded_layer_params.get(example_layer)
# if padded_tensors_example:
#     print(f"Padded tensors for {example_layer}:")
#     for i, tensor in enumerate(padded_tensors_example):
#         print(f"Tensor {i}: shape {tensor.shape}")
# else:
#     print(f"Layer {example_layer} not found.")

# %%
# Sample state_dict structure for demonstration purposes


def plot_and_save_state_dict(state_dict, save_path):
    """
    Plots the weights and biases from a state dictionary and saves the plot to the specified path.

    Parameters:
    - state_dict: dict, contains the state dictionary with layers' weights and biases.
    - save_path: Pathlib Path, the path where the plot image will be saved.
    """
    def plot_tensor_heatmap(ax, tensor, title="Tensor Heatmap", height=5, aspect='equal', repeat_factor=1):
        """
        Plots a 1D or 2D tensor as a heatmap on a given axis with a specified height.
        """
        array = tensor.numpy()
        if len(array.shape) == 1:
            # Reshape 1D tensor to 2D for horizontal plotting
            array = array.reshape(1, -1)
            # Repeat the array to make it repeat_factor x n
            array = array.repeat(repeat_factor, axis=0)

        # Calculate width to maintain aspect ratio
        fig_width = height * (array.shape[1] / array.shape[0])
        ax.set_aspect(aspect)
        ax.figure.set_size_inches(fig_width, height)

        cax = ax.imshow(array, aspect=aspect, cmap='viridis')
        ax.set_title(title)
        plt.colorbar(cax, ax=ax)

    # Define the layers for the encoder and decoder
    encoder_layers = ['encoder.0', 'encoder.1', 'encoder.2', 'encoder.3']
    decoder_layers = ['decoder.3', 'decoder.2', 'decoder.1', 'decoder.0']

    # Create a figure with subplots arranged in two rows and double the columns
    fig, axs = plt.subplots(2, len(encoder_layers) * 2, figsize=(20, 10))

    # Plot the encoder layers in the first row
    for i, layer in enumerate(encoder_layers):
        weight_key = f"{layer}.weight"
        bias_key = f"{layer}.bias"

        plot_tensor_heatmap(axs[0, i*2], state_dict[weight_key],
                            title=f"{weight_key} Heatmap", height=5)
        plot_tensor_heatmap(axs[0, i*2 + 1], state_dict[bias_key],
                            title=f"{bias_key} Heatmap", height=5, repeat_factor=10)

    # Plot the decoder layers in reverse order in the second row
    for i, layer in enumerate(decoder_layers):
        weight_key = f"{layer}.weight"
        bias_key = f"{layer}.bias"

        plot_tensor_heatmap(axs[1, i*2], state_dict[weight_key],
                            title=f"{weight_key} Heatmap", height=5)
        plot_tensor_heatmap(axs[1, i*2 + 1], state_dict[bias_key],
                            title=f"{bias_key} Heatmap", height=5, repeat_factor=10)

    plt.tight_layout()

    # Save the figure to the specified path
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    # plt.show()

    # return fig
# %%


def plot_single_tensor(tensor, title="Tensor Heatmap", height=5, aspect='equal', repeat_factor=1, save_path=None):
    """
    Plots a single tensor as a heatmap and optionally saves the plot to a specified path.

    Parameters:
    - tensor: torch.Tensor, the tensor to be plotted.
    - title: str, the title of the plot.
    - height: int, the height of the plot.
    - aspect: str, the aspect ratio of the plot.
    - repeat_factor: int, how many times to repeat 1D tensor for visualization.
    - save_path: Pathlib Path or str, the path where the plot image will be saved. If None, the plot is not saved.
    """
    array = tensor.numpy()
    if len(array.shape) == 1:
        # Reshape 1D tensor to 2D for horizontal plotting
        array = array.reshape(1, -1)
        # Repeat the array to make it repeat_factor x n
        array = array.repeat(repeat_factor, axis=0)

    # Calculate width to maintain aspect ratio
    fig_width = height * (array.shape[1] / array.shape[0])
    fig, ax = plt.subplots(figsize=(fig_width, height))
    ax.set_aspect(aspect)

    cax = ax.imshow(array, aspect=aspect, cmap='viridis')
    ax.set_title(title)
    plt.colorbar(cax, ax=ax)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()
    return fig


# %%
sizes_max = []
for key, value in padded_layer_params.items():
    sizes_max.append(value[-1].size())
for i in range(len(data)):
    path = f'heatmaps/state_dict_heatmap{i}.png'
    state_dict = {
        'encoder.0.weight': padded_layer_params['encoder.0.weight'][i],
        'encoder.0.bias': padded_layer_params['encoder.0.bias'][i],
        'encoder.1.weight': padded_layer_params['encoder.1.weight'][i],
        'encoder.1.bias': padded_layer_params['encoder.1.bias'][i],
        'encoder.2.weight': padded_layer_params['encoder.2.weight'][i],
        'encoder.2.bias': padded_layer_params['encoder.2.bias'][i],
        'encoder.3.weight': padded_layer_params['encoder.3.weight'][i],
        'encoder.3.bias': padded_layer_params['encoder.3.bias'][i],
        'decoder.0.weight': padded_layer_params['decoder.0.weight'][i],
        'decoder.0.bias': padded_layer_params['decoder.0.bias'][i],
        'decoder.1.weight': padded_layer_params['decoder.1.weight'][i],
        'decoder.1.bias': padded_layer_params['decoder.1.bias'][i],
        'decoder.2.weight': padded_layer_params['decoder.2.weight'][i],
        'decoder.2.bias': padded_layer_params['decoder.2.bias'][i],
        'decoder.3.weight': padded_layer_params['decoder.3.weight'][i],
        'decoder.3.bias': padded_layer_params['decoder.3.bias'][i],
    }
    plot_and_save_state_dict(state_dict, path)

# %%


def create_gif_from_images(image_folder, gif_path, duration=500):
    """
    Converts a folder of images into a GIF.

    Parameters:
    - image_folder: str or Path, path to the folder containing images.
    - gif_path: str or Path, path where the GIF will be saved.
    - duration: int, duration in milliseconds for each frame in the GIF.
    """
    # Ensure the folder and gif_path are Path objects
    image_folder = Path(image_folder)
    gif_path = Path(gif_path)

    # Collect all image file paths
    # Assuming images are in PNG format
    image_files = sorted(image_folder.glob('*.png'))
    if not image_files:
        raise ValueError("No images found in the provided folder.")

    # Open images and collect them in a list
    images = [Image.open(image_file) for image_file in image_files]

    # Save images as a GIF
    images[0].save(gif_path, save_all=True,
                   append_images=images[1:], duration=duration, loop=0)


# Example usage:
image_folder = r'C:\Users\Admin\Documents\GitHub\Neurogenesis\Modelweighs\Parameters_20240828_222648\heatmaps'
gif_path = r'C:\Users\Admin\Documents\GitHub\Neurogenesis\Modelweighs\Parameters_20240828_222648\output.gif'
create_gif_from_images(image_folder, gif_path, duration=500)
# %%
