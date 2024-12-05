
# %%#%%
# Imports and Dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, TensorDataset
import torch.nn.functional as F
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.optim.lr_scheduler import ExponentialLR
import math
import numpy as np
from collections import Counter
from typing import Union
import pandas as pd
from datetime import datetime
import torchvision.utils
from sklearn.cluster import KMeans
import random
from torch.utils.tensorboard import SummaryWriter
#%%
# custom losses
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        # Calculate the Euclidean distance between the two embeddings
        distance = F.pairwise_distance(embedding1, embedding2)

        # Contrastive Loss function
        loss_positive = label * torch.pow(distance, 2)  # For positive pairs
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)  # For negative pairs

        # Average the loss over the batch
        loss = 0.5 * (loss_positive + loss_negative).mean()
        return loss
# %%
# Model Definition
class NGAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, act_func=None, classification_head=None, num_classes_pretraining=None, classification_size=None, dropout=None):
        """
        Initializes the NGAutoencoder.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_sizes (list): List of hidden layer sizes.
            act_func (callable, optional): Activation function to use after each layer.
                If None, no activation function is applied. Defaults to F.relu.
        """
        super(NGAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes.copy()
        self.act_func = act_func
        self.classification_size=classification_size or 15
        self.classification=False
        self.return_encoded = False
        self.classification_head=classification_head or False
        self.num_classes_pretraining= num_classes_pretraining or None
        self.dropout = dropout or 0

        # Build encoder and decoder layers
        self.encoder_layers = self.build_encoder()
        self.decoder_layers = self.build_decoder()
        self.classifier = self.build_classifier()

        # Create Sequential models
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        # Store initial layer sizes
        self.initial_layer_sizes = self.get_layer_sizes()

        # Initialize a list to track layer sizes over time
        self.layer_size_history = [self.hidden_sizes.copy()]

    def build_encoder(self):
        layers = []
        in_dim = self.input_dim
        num_layers = len(self.hidden_sizes)
        for i, out_dim in enumerate(self.hidden_sizes):
            layer = NGLinear(in_dim, out_features_old=out_dim)
            layers.append(layer)
            if i < num_layers - 1:
                dropout_layer = nn.Dropout(self.dropout)
                layers.append(dropout_layer)
            in_dim = out_dim
        return layers

    def build_decoder(self):
        layers = []
        reversed_hidden_sizes = list(reversed(self.hidden_sizes))
        in_dim = reversed_hidden_sizes[0]
        num_layers = len(reversed_hidden_sizes)
        for i, out_dim in enumerate(reversed_hidden_sizes[1:]):
            layer = NGLinear(in_dim, out_features_old=out_dim)
            layers.append(layer)
            dropout_layer = nn.Dropout(self.dropout)
            layers.append(dropout_layer)
            in_dim = out_dim
        # Final decoder layer should map back to input_dim
        layer = NGLinear(in_dim, out_features_old=self.input_dim)
        layers.append(layer)
        return layers

    def build_classifier(self):
                # Classification head
        classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], self.classification_size),
            nn.ReLU(True),
            nn.Linear(self.classification_size, self.num_classes_pretraining)
        )
        return classifier
    

    def get_layer_sizes(self):
        sizes = []
        for layer in self.encoder_layers + self.decoder_layers:
            if isinstance(layer, NGLinear):
                sizes.append({'in_features': layer.in_features, 'out_features': layer.out_features})
        return sizes

    def forward(self, x, l=None):
        # If l is None, use the full network
        if l is None:
            l = len(self.encoder_layers)
        else:
            l = l + 1  # Adjust l to include up to layer l
            if l > len(self.encoder_layers):
                l = len(self.encoder_layers)
        # Encode with first l layers
        encoded = x
        for layer in self.encoder_layers[:l]:
            encoded = layer(encoded)
            if self.act_func is not None:
                encoded = self.act_func(encoded)
        # Decode with corresponding layers
        decoder_start_idx = len(self.decoder_layers) - l
        decoded = encoded
        for layer in self.decoder_layers[decoder_start_idx:]:
            decoded = layer(decoded)
            if self.act_func is not None:
                decoded = self.act_func(decoded)
        if self.return_encoded: return encoded, decoded
        elif self.classification:
            class_logits = self.classifier(encoded)
            return decoded, class_logits
        else: return decoded

    def add_nodes(self, layer_idx, num_new_nodes):
        # Update hidden sizes
        self.hidden_sizes[layer_idx] += num_new_nodes

        # Add new nodes to the specified layer in encoder
        self.encoder_layers[layer_idx].add_new_nodes(num_new_nodes)

        # Adjust input size of the next encoder layer
        if layer_idx + 1 < len(self.encoder_layers):
            self.encoder_layers[layer_idx + 1].adjust_input_size(num_new_nodes)

        # Adjust the decoder layers to mirror the encoder changes
        decoder_layer_idx = len(self.decoder_layers) - layer_idx - 1

        # Adjust input size of the decoder layer corresponding to encoder layer
        self.decoder_layers[decoder_layer_idx].adjust_input_size(num_new_nodes)

        # Adjust output size of the previous decoder layer
        if decoder_layer_idx - 1 >= 0:
            self.decoder_layers[decoder_layer_idx - 1].add_new_nodes(num_new_nodes)

        # Rebuild encoder and decoder
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        # Update initial layer sizes
        self.initial_layer_sizes = self.get_layer_sizes()

        # Record the updated hidden sizes
        self.layer_size_history.append(self.hidden_sizes.copy())

    def get_new_params(self, module='model'):
        new_params = []
        if module == 'encoder' or module == 'model':
            for layer in self.encoder_layers:
                if layer.weight_new is not None:
                    new_params.append(layer.weight_new)
                    new_params.append(layer.bias_new)
        if module == 'decoder' or module == 'model':
            for layer in self.decoder_layers:
                if layer.weight_new is not None:
                    new_params.append(layer.weight_new)
                    new_params.append(layer.bias_new)
        return new_params

    def get_old_params(self, module='model'):
        old_params = []
        if module == 'encoder' or module == 'model':
            if not self.encoder_layers:
                print("Warning: Encoder layers are empty.")
            for layer in self.encoder_layers:
                if hasattr(layer, 'weight_old') and hasattr(layer, 'bias_old'):
                    old_params.append(layer.weight_old)
                    old_params.append(layer.bias_old)
                else:
                    print(f"Warning: Layer {layer} does not have 'weight_old' and 'bias_old' attributes.")
        if module == 'decoder' or module == 'model':
            if not self.decoder_layers:
                print("Warning: Decoder layers are empty.")
            for layer in self.decoder_layers:
                if hasattr(layer, 'weight_old') and hasattr(layer, 'bias_old'):
                    old_params.append(layer.weight_old)
                    old_params.append(layer.bias_old)
                else:
                    print(f"Warning: Layer {layer} does not have 'weight_old' and 'bias_old' attributes.")
        return old_params

    def freeze_layers(self, layers):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_layers(self, layers):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    def consolidate_new_nodes(self):
        """
        Consolidate all new nodes' weights and biases as old ones, and reset new ones.
        """
        for layer in self.encoder_layers + self.decoder_layers:
            if isinstance(layer, NGLinear):
                layer.promote_new_to_old()

    def remove_dropout_from_sequential(self, sequential):
        """Remove all Dropout layers from a Sequential module."""
        layers = [layer for layer in sequential if not isinstance(layer, nn.Dropout)]
        return nn.Sequential(*layers)

    def remove_all_dropout(self):
        """Remove all Dropout layers from the encoder, decoder, and their layers."""
        def remove_dropout_from_list(layers):
            return [layer for layer in layers if not isinstance(layer, nn.Dropout)]
        
        # Update encoder layers and Sequential
        self.encoder_layers = remove_dropout_from_list(self.encoder_layers)
        self.encoder = nn.Sequential(*self.encoder_layers)

        # Update decoder layers and Sequential
        self.decoder_layers = remove_dropout_from_list(self.decoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        print("All Dropout layers have been removed.")

# Custom Linear Layer
class NGLinear(nn.Module):
    def __init__(self, in_features, out_features_old, out_features_new=0):
        super(NGLinear, self).__init__()
        self.in_features = in_features
        self.out_features_old = out_features_old
        self.out_features_new = out_features_new

        # Old parameters
        self.weight_old = nn.Parameter(torch.Tensor(out_features_old, in_features))
        self.bias_old = nn.Parameter(torch.Tensor(out_features_old))

        # New parameters
        if out_features_new > 0:
            self.weight_new = nn.Parameter(torch.Tensor(out_features_new, in_features))
            self.bias_new = nn.Parameter(torch.Tensor(out_features_new))
        else:
            self.weight_new = None
            self.bias_new = None

        self.reset_parameters()

    @property
    def out_features(self):
        return self.out_features_old + self.out_features_new

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_old, a=math.sqrt(5))
        nn.init.zeros_(self.bias_old)
        if self.weight_new is not None:
            nn.init.kaiming_uniform_(self.weight_new, a=math.sqrt(5))
            nn.init.zeros_(self.bias_new)

    def forward(self, input):
        output_old = F.linear(input, self.weight_old, self.bias_old)
        if self.weight_new is not None:
            output_new = F.linear(input, self.weight_new, self.bias_new)
            output = torch.cat((output_old, output_new), dim=1)
        else:
            output = output_old
        return output

    def add_new_nodes(self, num_new_nodes):
        # Initialize new weights and biases
        new_weight = nn.Parameter(torch.Tensor(num_new_nodes, self.in_features))
        nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

        new_bias = nn.Parameter(torch.Tensor(num_new_nodes))
        nn.init.zeros_(new_bias)

        # Concatenate new parameters
        if self.weight_new is not None:
            self.weight_new = nn.Parameter(torch.cat([self.weight_new, new_weight], dim=0))
            self.bias_new = nn.Parameter(torch.cat([self.bias_new, new_bias], dim=0))
        else:
            self.weight_new = new_weight
            self.bias_new = new_bias

        self.out_features_new += num_new_nodes

    def adjust_input_size(self, num_new_inputs):
        # Adjust in_features
        self.in_features += num_new_inputs

        # Initialize new weights
        new_weights_old = torch.Tensor(self.out_features_old, num_new_inputs)
        nn.init.kaiming_uniform_(new_weights_old, a=math.sqrt(5))
        self.weight_old = nn.Parameter(torch.cat([self.weight_old, new_weights_old], dim=1))

        # Expand weight_new if it exists
        if self.weight_new is not None:
            new_weights_new = torch.Tensor(self.out_features_new, num_new_inputs)
            nn.init.kaiming_uniform_(new_weights_new, a=math.sqrt(5))
            self.weight_new = nn.Parameter(torch.cat([self.weight_new, new_weights_new], dim=1))

    def promote_new_to_old(self):
        """
        Consolidate new weights and biases into old ones, and reset new weights and biases.
        """
        if self.weight_new is not None:
            # Concatenate new weights to old weights
            self.weight_old = nn.Parameter(torch.cat([self.weight_old, self.weight_new], dim=0))
            self.bias_old = nn.Parameter(torch.cat([self.bias_old, self.bias_new], dim=0))
            # Reset new weights and biases
            self.weight_new = None
            self.bias_new = None
            # Update out_features_old and reset out_features_new
            self.out_features_old += self.out_features_new
            self.out_features_new = 0
#%%
# Data Utilities
def filter_by_class(dataset, classes):
    indices = [i for i, (img, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)

def count_samples_per_class(dataset):
    class_counts = {}
    for _, label in dataset:
        label = label.item() if isinstance(label, torch.Tensor) else label
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    return class_counts

def average_dict_values(data_dict):
    total = sum(data_dict.values())
    count = len(data_dict)
    average = total / count if count > 0 else 0
    return average

class SyntheticMNIST(Dataset):
    def __init__(self, data, labels):
        self.data = data  # List of tensors (images)
        self.labels = labels  # Corresponding labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if idx < 0 or idx >= len(self.data):
        #     raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.data)}")
        return self.data[idx], self.labels[idx]

# Function to compute class statistics with Cholesky decomposition
def get_class_stats_with_cholesky(encoder, data_loader):
    encoder.eval()
    class_predictions = {}

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(images.size(0), -1)
            outputs = encoder(images)
            for label, output in zip(labels, outputs):
                label = label.item()
                if label not in class_predictions:
                    class_predictions[label] = []
                class_predictions[label].append(output.cpu().numpy())

    class_stats = {}
    for label, predictions in class_predictions.items():
        predictions = np.stack(predictions)
        mean = np.mean(predictions, axis=0)
        # Calculate the covariance matrix
        epsilon = 1e-6  # Small jitter value
        cov_matrix = np.cov(predictions, rowvar=False)
        cov_matrix += epsilon * np.eye(cov_matrix.shape[0])  # Add jitter to diagonal, to increace probability of poditive definte matrix
        # Cholesky decomposition of the covariance matrix
        try:
            chol_cov = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # If the covariance matrix is not positive-definite, add a small value to the diagonal
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            chol_cov = np.linalg.cholesky(cov_matrix)
        class_stats[label] = {'mean': mean, 'cholesky': chol_cov}
    return class_stats

# Function to sample from class statistics using Cholesky factor
def sample_from_stats_cholesky(class_stats, num_samples_per_class):
    sampled_outputs = {}
    for class_label, stats in class_stats.items():
        mean = stats['mean']
        chol_cov = stats['cholesky']
        # Sample from a standard normal distribution
        z = np.random.randn(num_samples_per_class, mean.shape[0])
        # Transform the samples
        samples = z @ chol_cov.T + mean
        sampled_outputs[class_label] = samples
    return sampled_outputs

# Function to create synthetic dataset
def create_synthetic_dataset(encoder, decoder, data_loader, num_samples_per_class):
    """
    Generates a synthetic dataset using the intrinsic replay method.
    """
    class_stats = get_class_stats_with_cholesky(encoder, data_loader)
    sampled_data = sample_from_stats_cholesky(class_stats, num_samples_per_class)

    synthetic_images = []
    synthetic_labels = []
    decoder.eval()
    with torch.no_grad():
        for label, samples in sampled_data.items():
            samples = torch.tensor(samples).float()
            images = decoder(samples)
            images = images.view(-1, 1, 28, 28)  # Adjust based on your data shape
            synthetic_images.extend(images)
            synthetic_labels.extend([label] * len(images))
    synthetic_dataset = SyntheticMNIST(synthetic_images, synthetic_labels)
    return synthetic_dataset

#%%
# Training Utilities
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, rc_n_samples=4, pretrained_classes=[], get_samples_and_reconstructions=None, log_samples_to_tensorboard=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_losses = []
        self.test_losses = []
        self.rc_n_samples = rc_n_samples
        self.pretrained_classes = pretrained_classes
        self.get_samples_and_reconstructions = get_samples_and_reconstructions
        self.log_samples_to_tensorboard = log_samples_to_tensorboard

    def train(self, train_loader, test_loader=None, epochs=1):
        epoch_bar = tqdm(range(epochs), desc='Training Progress')
        for epoch in epoch_bar:
            self.model.train()
            epoch_losses = []
            batch_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
            for data, _ in batch_bar:
                data = data.view(data.size(0), -1)  # Flatten the images
                output = self.model(data)
                loss = self.criterion(output, data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                batch_bar.set_postfix(loss=loss.item())
            if self.scheduler:
                writer.add_scalars('Learning Rate Schedule', {'Learning Rate': self.scheduler.get_last_lr()[-1]}, epoch + 1)
                self.scheduler.step()

            training_loss = sum(epoch_losses) / len(epoch_losses)
            self.training_losses.append(training_loss)

            if test_loader:
                test_loss = self.test(test_loader)
                self.test_losses.append(test_loss)
                #self.plot_losses(epoch + 1)
                writer.add_scalars('Loss_Pretraining', {'Training': training_loss, 'Test': test_loss}, epoch + 1)
                
                if epoch % 10 == 0:
                    for c in self.pretrained_classes:
                        # Get the images
                        original_imgs, reconstructed_imgs, synthetic_imgs = self.get_samples_and_reconstructions(c, self.rc_n_samples, synthetic=False)
                        # Log them to TensorBoard
                        self.log_samples_to_tensorboard(original_imgs, reconstructed_imgs, synthetic_imgs, c, epoch)

    def test(self, test_loader, l=None):
        self.model.eval()
        total_loss = 0
        batch_bar = tqdm(test_loader, desc='Testing', leave=False)
        with torch.no_grad():
            for data, _ in batch_bar:
                data = data.view(data.size(0), -1)
                output = self.model(data, l)
                loss = self.criterion(output, data)
                total_loss += loss.item()
                batch_bar.set_postfix(loss=loss.item())
        average_loss = total_loss / len(test_loader)
        return average_loss

    def plot_losses(self, epoch):
        clear_output(wait=True)
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title(f'Training and Test Loss Progress: Epoch {epoch}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot layer size development
        plt.subplot(1, 2, 2)
        layer_size_history = np.array(self.model.layer_size_history)
        for i in range(layer_size_history.shape[1]):
            plt.plot(layer_size_history[:, i], label=f'Layer {i+1}')
        plt.title('Layer Sizes Over Time')
        plt.xlabel('Neurogenesis Steps')
        plt.ylabel('Number of Nodes')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class ClusteredTrainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, alpha=1.0, n_clusters=2, writer=None):
        self.model = model
        self.criterion = criterion  # Reconstruction loss criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.alpha = alpha  # Weighting factor for clustering loss
        self.n_clusters = n_clusters  # Number of clusters/classes
        self.training_losses = []
        self.test_losses = []
        self.training_reconstruction_losses = []
        self.training_clustering_losses = []
        self.test_reconstruction_losses = []
        self.test_clustering_losses = []
        self.device = next(model.parameters()).device  # Device (CPU or GPU)
        self.cluster_centers = None  # Will be initialized later
        self.writer = writer  # TensorBoard writer

    def initialize_cluster_centers(self, data_loader):
        # Extract latent representations
        all_latents = []
        self.model.eval()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.view(data.size(0), -1).to(self.device)
                latent = self.model.encoder(data)
                all_latents.append(latent)
        all_latents = torch.cat(all_latents).cpu().numpy()

        # Perform KMeans clustering on latent space to initialize cluster centers
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(all_latents)
        self.cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

    def update_cluster_centers(self, data_loader):
        # Update cluster centers based on new latent representations
        self.initialize_cluster_centers(data_loader)

    def target_distribution(self, q):
        weight = (q ** 2) / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def train(self, train_loader, test_loader=None, epochs=1, cluster_update_interval=1):
        # Initialize cluster centers
        self.initialize_cluster_centers(train_loader)

        epoch_bar = tqdm(range(epochs), desc='Training Progress')
        for epoch in epoch_bar:
            self.model.train()
            epoch_reconstruction_losses = []
            epoch_clustering_losses = []
            epoch_combined_losses = []
            batch_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
            for data, _ in batch_bar:
                data = data.view(data.size(0), -1).to(self.device)
                # Forward pass
                latent = self.model.encoder(data)
                output = self.model.decoder(latent)
                # Reconstruction loss
                reconstruction_loss = self.criterion(output, data)
                # Clustering loss
                q = 1.0 / (1.0 + torch.cdist(latent, self.cluster_centers, p=2) ** 2)
                q = (q.T / q.sum(1)).T  # Normalize per sample
                # Compute target distribution
                p = self.target_distribution(q)
                # KL divergence loss
                kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
                # Total loss
                loss = reconstruction_loss + self.alpha * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_reconstruction_losses.append(reconstruction_loss.item())
                epoch_clustering_losses.append(kl_loss.item())
                epoch_combined_losses.append(loss.item())
                batch_bar.set_postfix(loss=loss.item())
            lr_old = self.optimizer.lern
            if self.scheduler:
                self.scheduler.step()

            # Log average losses
            avg_reconstruction_loss = sum(epoch_reconstruction_losses) / len(epoch_reconstruction_losses)
            avg_clustering_loss = sum(epoch_clustering_losses) / len(epoch_clustering_losses)
            avg_combined_loss = sum(epoch_combined_losses) / len(epoch_combined_losses)
            self.training_losses.append(avg_combined_loss)
            self.training_reconstruction_losses.append(avg_reconstruction_loss)
            self.training_clustering_losses.append(avg_clustering_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Reconstruction Loss: {avg_reconstruction_loss:.4f}, Clustering Loss: {avg_clustering_loss:.4f}, Combined Loss: {avg_combined_loss:.4f}")

            # Update cluster centers periodically
            if (epoch + 1) % cluster_update_interval == 0:
                self.update_cluster_centers(train_loader)

            # Optionally, evaluate on test_loader
            if test_loader:
                test_reconstruction_loss, test_clustering_loss, test_combined_loss = self.test(test_loader)
                self.test_losses.append(test_combined_loss)
                self.test_reconstruction_losses.append(test_reconstruction_loss)
                self.test_clustering_losses.append(test_clustering_loss)
                # Log test losses
                if self.writer is not None:
                    self.writer.add_scalars('Loss/Combined', {'Training': avg_combined_loss, 'Test': test_combined_loss}, epoch + 1)
                    self.writer.add_scalars('Loss/Reconstruction', {'Training': avg_reconstruction_loss, 'Test': test_reconstruction_loss}, epoch + 1)
                    self.writer.add_scalars('Loss/Clustering', {'Training': avg_clustering_loss, 'Test': test_clustering_loss}, epoch + 1)
            else:
                if self.writer is not None:
                    self.writer.add_scalar('Loss/Combined/Training', avg_combined_loss, epoch + 1)
                    self.writer.add_scalar('Loss/Reconstruction/Training', avg_reconstruction_loss, epoch + 1)
                    self.writer.add_scalar('Loss/Clustering/Training', avg_clustering_loss, epoch + 1)

    def test(self, test_loader):
        self.model.eval()
        total_reconstruction_loss = 0
        total_clustering_loss = 0
        total_combined_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.view(data.size(0), -1).to(self.device)
                latent = self.model.encoder(data)
                output = self.model.decoder(latent)
                reconstruction_loss = self.criterion(output, data)
                q = 1.0 / (1.0 + torch.cdist(latent, self.cluster_centers, p=2) ** 2)
                q = (q.T / q.sum(1)).T  # Normalize per sample
                # Compute target distribution
                p = self.target_distribution(q)
                # KL divergence loss
                kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
                # Total loss
                loss = reconstruction_loss + self.alpha * kl_loss

                total_reconstruction_loss += reconstruction_loss.item()
                total_clustering_loss += kl_loss.item()
                total_combined_loss += loss.item()

        average_reconstruction_loss = total_reconstruction_loss / len(test_loader)
        average_clustering_loss = total_clustering_loss / len(test_loader)
        average_combined_loss = total_combined_loss / len(test_loader)
        return average_reconstruction_loss, average_clustering_loss, average_combined_loss

class ClassificationTrainer:
    def __init__(self, model, recon_criterion, class_criterion, classes_pretraining, optimizer, scheduler=None, alpha=1.0):
        """
        Initializes the trainer with reconstruction and classification criteria.
        
        :param model: The autoencoder model with classification head.
        :param recon_criterion: Loss function for reconstruction (e.g., nn.MSELoss()).
        :param class_criterion: Loss function for classification (e.g., nn.CrossEntropyLoss()).
        :param optimizer: Optimizer for training.
        :param scheduler: Learning rate scheduler.
        :param alpha: Weighting factor between reconstruction and classification losses.
        """
        self.model = model
        self.recon_criterion = recon_criterion
        self.class_criterion = class_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_losses = []
        self.test_losses = []
        self.alpha = alpha  # Weight for classification loss
        self.class_map = {label: idx for idx, label in enumerate(classes_pretraining)}

    def train(self, train_loader, test_loader=None, epochs=1, writer=None):
        epoch_bar = tqdm(range(epochs), desc='Training Progress')
        for epoch in epoch_bar:
            self.model.train()
            epoch_recon_losses = []
            epoch_class_losses = []
            batch_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
            for data, labels in batch_bar:
                data = data.view(data.size(0), -1)  # Flatten the images
                labels = torch.tensor([self.class_map[label.item()] for label in labels])
                labels = labels.long()

                # Forward pass
                reconstruction, class_logits = self.model(data)
                
                # Compute losses
                recon_loss = self.recon_criterion(reconstruction, data)
                class_loss = self.class_criterion(class_logits, labels)
                total_loss = (recon_loss) + (self.alpha * class_loss)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Record losses
                epoch_recon_losses.append(recon_loss.item())
                epoch_class_losses.append(class_loss.item())
                
                # Update progress bar
                batch_bar.set_postfix(recon_loss=recon_loss.item(), class_loss=class_loss.item())
            
            if self.scheduler:
                self.scheduler.step()
            
            # Calculate average losses for the epoch
            avg_recon_loss = sum(epoch_recon_losses) / len(epoch_recon_losses)
            avg_class_loss = sum(epoch_class_losses) / len(epoch_class_losses)
            self.training_losses.append((avg_recon_loss, avg_class_loss))
            
            if test_loader:
                test_recon_loss, test_class_loss, cm = self.test(test_loader)
                self.test_losses.append((test_recon_loss, test_class_loss))
                if writer:
                    writer.add_scalars('Pretraining', {
                        'Training_Reconstruction': avg_recon_loss,
                        'Training_Classification': avg_class_loss,
                        'Test_Reconstruction': test_recon_loss,
                        'Test_Classification': test_class_loss
                    }, epoch + 1)
                            # Add confusion matrix to TensorBoard
                class_names = [str(c) for c in self.class_map.values()]
                cm_fig = self.plot_confusion_matrix(cm, class_names)
                writer.add_figure('Confusion Matrix', cm_fig, global_step=epoch + 1)
                    
            # Update epoch description
            epoch_bar.set_postfix(
                Training_Recon_Loss=avg_recon_loss,
                Training_Class_Loss=avg_class_loss,
                Test_Recon_Loss=test_recon_loss if test_loader else 'N/A',
                Test_Class_Loss=test_class_loss if test_loader else 'N/A'
            )
    
    def test(self, test_loader):
        self.model.eval()
        total_recon_loss = 0
        total_class_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc='Testing', leave=False):
                data = data.view(data.size(0), -1)
                labels = torch.tensor([self.class_map[label.item()] for label in labels])
                labels = labels.long()
                reconstruction, class_logits = self.model(data)
                recon_loss = self.recon_criterion(reconstruction, data)
                class_loss = self.class_criterion(class_logits, labels)
                total_recon_loss += recon_loss.item()
                total_class_loss += class_loss.item()

                # Collect predictions and true labels for confusion matrix
                _, preds = torch.max(class_logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        average_recon_loss = total_recon_loss / len(test_loader)
        average_class_loss = total_class_loss / len(test_loader)
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        return average_recon_loss, average_class_loss, cm

    def plot_confusion_matrix(self, cm, class_names):
        import matplotlib.pyplot as plt
        import numpy as np
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Add text in each cell
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

class ContrastTrainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, alpha=1.0, writer=None, batch_size=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.alpha = alpha  # Weighting factor for contrastive loss
        self.writer = writer  # TensorBoard writer
        self.training_losses = []
        self.test_losses = []
        self.batch_size = batch_size or 32
        self.device = next(model.parameters()).device  # Device (CPU or GPU)

    def create_pairs(self, data, labels):
        # Generate positive and negative pairs
        pairs = []
        pair_labels = []
        for i in range(len(data)):
            current_data = data[i]
            current_label = labels[i]

            # Positive pair
            positive_idx = random.choice([idx for idx, lbl in enumerate(labels) if lbl == current_label and idx != i])
            positive_data = data[positive_idx]
            pairs.append((current_data, positive_data))
            pair_labels.append(1)  # Label 1 for positive pairs

            # Negative pair
            negative_idx = random.choice([idx for idx, lbl in enumerate(labels) if lbl != current_label])
            negative_data = data[negative_idx]
            pairs.append((current_data, negative_data))
            pair_labels.append(0)  # Label 0 for negative pairs
        
        return pairs, pair_labels

    def train(self, train_loader, test_loader=None, epochs=1):
        epoch_bar = tqdm(range(epochs), desc='Training Progress')
        for epoch in epoch_bar:
            self.model.train()
            epoch_reconstruction_losses = []
            epoch_contrastive_losses = []
            epoch_combined_losses = []

            # Create pairs and labels for contrastive learning
            for data, labels in train_loader:
                pairs, pair_labels = self.create_pairs(data, labels)
                pair_data = TensorDataset(
                    torch.stack([p[0] for p in pairs]), 
                    torch.stack([p[1] for p in pairs]), 
                    torch.tensor(pair_labels)
                )
                pair_loader = DataLoader(pair_data, batch_size=self.batch_size, shuffle=True)

                # Training loop for pairwise data
                batch_bar = tqdm(pair_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
                for data1, data2, labels in batch_bar:
                    data1 = data1.view(data1.size(0), -1).to(self.device)
                    data2 = data2.view(data2.size(0), -1).to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    # latent1 = self.model.encoder(data1)
                    # latent2 = self.model.encoder(data2)
                    # output1 = self.model.decoder(latent1)
                    # output2 = self.model.decoder(latent2)

                    latent1, output1 = self.model(data1)
                    latent2, output2 = self.model(data2)

                    # Reconstruction loss
                    reconstruction_loss = self.criterion(output1, data1) + self.criterion(output2, data2)
                    
                    # Contrastive loss
                    contrastive_loss = ContrastiveLoss(margin=1.0)(latent1, latent2, labels)

                    # Total loss
                    loss = reconstruction_loss + self.alpha * contrastive_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_reconstruction_losses.append(reconstruction_loss.item())
                    epoch_contrastive_losses.append(contrastive_loss.item())
                    epoch_combined_losses.append(loss.item())
                    batch_bar.set_postfix(loss=loss.item())

            if self.scheduler:
                self.scheduler.step()

            # Log average losses
            avg_reconstruction_loss = sum(epoch_reconstruction_losses) / len(epoch_reconstruction_losses)
            avg_contrastive_loss = sum(epoch_contrastive_losses) / len(epoch_contrastive_losses)
            avg_combined_loss = sum(epoch_combined_losses) / len(epoch_combined_losses)
            self.training_losses.append(avg_combined_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Reconstruction Loss: {avg_reconstruction_loss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}, Combined Loss: {avg_combined_loss:.4f}")

            # if self.writer is not None:
            #     self.writer.add_scalar('Loss/Combined/Training', avg_combined_loss, epoch + 1)
            #     self.writer.add_scalar('Loss/Reconstruction/Training', avg_reconstruction_loss, epoch + 1)
            #     self.writer.add_scalar('Loss/Contrastive/Training', avg_contrastive_loss, epoch + 1)

            # # Evaluate on test set if available
            # if test_loader:
            #     test_reconstruction_loss, test_contrastive_loss, test_combined_loss = self.test(test_loader)
            #     self.test_losses.append(test_combined_loss)
            #     if self.writer is not None:
            #         self.writer.add_scalar('Loss/Combined/Test', test_combined_loss, epoch + 1)
            #         self.writer.add_scalar('Loss/Reconstruction/Test', test_reconstruction_loss, epoch + 1)
            #         self.writer.add_scalar('Loss/Contrastive/Test', test_contrastive_loss, epoch + 1)

            if test_loader:
                test_reconstruction_loss, test_contrastive_loss, test_combined_loss = self.test(test_loader)
                self.test_losses.append(test_combined_loss)

            if self.writer is not None:
                # Log both training and test loss for combined, reconstruction, and contrastive losses
                self.writer.add_scalars('Loss/Combined', {
                    'Training': avg_combined_loss,
                    'Test': test_combined_loss if test_loader else None
                }, epoch + 1)
                
                self.writer.add_scalars('Loss/Reconstruction', {
                    'Training': avg_reconstruction_loss,
                    'Test': test_reconstruction_loss if test_loader else None
                }, epoch + 1)
                
                self.writer.add_scalars('Loss/Contrastive', {
                    'Training': avg_contrastive_loss,
                    'Test': test_contrastive_loss if test_loader else None
                }, epoch + 1)
                
                for name, param in model.named_parameters():
                    if "weight" in name:
                        # Normalize weights for visualization
                        weight_matrix = param.data.cpu().numpy()
                        weight_matrix = (weight_matrix - weight_matrix.min()) / (weight_matrix.max() - weight_matrix.min())
                        
                        # Convert to a 3-channel grayscale image
                        weight_image = torch.tensor(weight_matrix).unsqueeze(0)  # Shape: [1, H, W]
                        weight_image = transforms.ToPILImage()(weight_image)  # Convert to PIL image

                        # Add to TensorBoard
                        self.writer.add_image(f"Weights/{name}", transforms.ToTensor()(weight_image), epoch)

    def test(self, test_loader):
        self.model.eval()
        total_reconstruction_loss = 0
        total_contrastive_loss = 0
        total_combined_loss = 0
        with torch.no_grad():
            for data, labels in test_loader:
                pairs, pair_labels = self.create_pairs(data, labels)
                pair_data = TensorDataset(
                    torch.stack([p[0] for p in pairs]), 
                    torch.stack([p[1] for p in pairs]), 
                    torch.tensor(pair_labels)
                )
                pair_loader = DataLoader(pair_data, batch_size=32, shuffle=False)

                for data1, data2, labels in pair_loader:
                    data1 = data1.view(data1.size(0), -1).to(self.device)
                    data2 = data2.view(data2.size(0), -1).to(self.device)
                    labels = labels.to(self.device)

                    latent1 = self.model.encoder(data1)
                    latent2 = self.model.encoder(data2)
                    output1 = self.model.decoder(latent1)
                    output2 = self.model.decoder(latent2)

                    # Reconstruction and contrastive losses
                    reconstruction_loss = self.criterion(output1, data1) + self.criterion(output2, data2)
                    contrastive_loss = ContrastiveLoss(margin=1.0)(latent1, latent2, labels)
                    loss = reconstruction_loss + self.alpha * contrastive_loss

                    total_reconstruction_loss += reconstruction_loss.item()
                    total_contrastive_loss += contrastive_loss.item()
                    total_combined_loss += loss.item()

        avg_reconstruction_loss = total_reconstruction_loss / len(test_loader)
        avg_contrastive_loss = total_contrastive_loss / len(test_loader)
        avg_combined_loss = total_combined_loss / len(test_loader)
        return avg_reconstruction_loss, avg_contrastive_loss, avg_combined_loss

#%%
# NGTrainer Class Definition
class NGTrainer:
    def __init__(self, model, train_dataset, test_dataset, writer, pretrained_classes=None, new_classes=None,
                 batch_size=None, lr=None, lr_factor=None, epochs=None, epochs_per_iteration=None,
                 thresholds=None, max_nodes=None, max_outliers=None, criterion=None, criterion_classification=None, factor_thr=None, factor_ng=None, metric_threshould=None, alpha=None, weight_decay=None):
        """
        Initializes the NGTrainer with the given parameters.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.pretrained_classes = pretrained_classes or []
        self.new_classes = new_classes or []
        self.batch_size = batch_size or 64
        self.lr = lr or 1e-3
        self.lr_factor = lr_factor or 1e-2
        self.epochs = epochs or 10
        self.epochs_per_iteration = epochs_per_iteration or 1
        self.max_nodes = max_nodes or [1500, 800, 500, 200]
        self.max_outliers = max_outliers or 5
        self.criterion = criterion or nn.MSELoss()
        self.criterion_classification = criterion_classification or nn.CrossEntropyLoss()
        self.factor_thr = factor_thr or 1.2
        self.neruons_per_new_class = {}
        self.factor_ng = factor_ng or 1e-1
        self.metric_threshould = metric_threshould or 'max'
        self.thresholds = thresholds
        self.writer = writer
        self.weight_decay = weight_decay or 10e-5
        # Initialize datasets
        self.init_datasets()
        self.alpha = alpha or 1


        # Initialize optimizer and scheduler
        self.init_optimizer_scheduler()

        # Initialize training history
        self.training_losses = []
        self.test_losses = []

        # Initialize list to track losses during neurogenesis
        self.losses_per_layer_history = []

    def init_datasets(self):
        # Filter datasets based on specified classes
        self.train_dataset_pretrained = filter_by_class(self.train_dataset, self.pretrained_classes)
        self.train_datasets_new = [filter_by_class(self.train_dataset, [c]) for c in self.new_classes]
        self.test_dataset_pretrained = filter_by_class(self.test_dataset, self.pretrained_classes)
        self.test_datasets_new = [filter_by_class(self.test_dataset, [c]) for c in self.new_classes]

        # Create data loaders
        self.train_loader_pretrained = DataLoader(self.train_dataset_pretrained, batch_size=self.batch_size, shuffle=True)
        self.train_loaders_new = [DataLoader(ds, batch_size=self.batch_size, shuffle=True) for ds in self.train_datasets_new]
        self.test_loader_pretrained = DataLoader(self.test_dataset_pretrained, batch_size=self.batch_size, shuffle=False)
        self.test_loaders_new = [DataLoader(ds, batch_size=self.batch_size, shuffle=False) for ds in self.test_datasets_new]

    def init_optimizer_scheduler(self):
        # Initialize optimizer for pretraining
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=1)

    def pretrain_model(self):
        """
        Pretrains the model on the specified pretrained classes.
        """
        print("Starting pretraining...")
        trainer = Trainer(self.model, self.criterion, self.optimizer, self.scheduler, pretrained_classes=self.pretrained_classes, get_samples_and_reconstructions=self.get_samples_and_reconstructions, log_samples_to_tensorboard=self.log_samples_to_tensorboard)
        trainer.train(self.train_loader_pretrained, self.test_loader_pretrained, epochs=self.epochs)
        self.training_losses.extend(trainer.training_losses)
        self.test_losses.extend(trainer.test_losses)
        loss_stats = self._compute_losses_per_layer_per_class(self.test_loader_pretrained)
        self.thresholds = self._get_thresholds_from_statistics_range(loss_stats=loss_stats, factor=self.factor_thr, metric=self.metric_threshould)
        self.model.remove_all_dropout()
        print("Pretraining completed.")

    def pretrain_model_clustered(self, alpha=1.0, cluster_update_interval=1):
        """
        Pretrains the model on the specified pretrained classes with clustering in the latent space.
        """
        print("Starting pretraining with clustering...")
        self.model.return_encoded = True
        trainer = ClusteredTrainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            alpha=alpha,
            n_clusters=len(self.pretrained_classes),
            writer=self.writer  # Pass the writer here
        )
        trainer.train(
            train_loader=self.train_loader_pretrained,
            test_loader=self.test_loader_pretrained,
            epochs=self.epochs,
            cluster_update_interval=cluster_update_interval
        )
        self.training_losses.extend(trainer.training_losses)
        self.test_losses.extend(trainer.test_losses)
        # Compute thresholds if needed
        loss_stats = self._compute_losses_per_layer_per_class(self.test_loader_pretrained)
        self.thresholds = self._get_thresholds_from_statistics_range(
            loss_stats=loss_stats,
            factor=self.factor_thr,
            metric=self.metric_threshould
        )
        self.model.return_encoded = False
        print("Pretraining with clustering completed.")

    def pretrain_model_classification(self):
        """
        Pretrains the model on the specified pretrained classes with both reconstruction and classification tasks.
        """
        print("Starting pretraining with classification...")
        self.model.classification = True
        # Initialize the trainer with both reconstruction and classification criteria
        trainer = ClassificationTrainer(
            model=self.model,
            recon_criterion=self.criterion,
            class_criterion=self.criterion_classification,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            alpha=self.alpha,  # Weighting factor for classification loss
            classes_pretraining=self.pretrained_classes
            )
        
        # Start training
        trainer.train(
            train_loader=self.train_loader_pretrained,
            test_loader=self.test_loader_pretrained,
            epochs=self.epochs,
            writer=self.writer
        )
        
        # Record losses
        self.training_losses.extend(trainer.training_losses)
        self.test_losses.extend(trainer.test_losses)
        
        # Compute thresholds based on reconstruction loss statistics
        loss_stats = self._compute_losses_per_layer_per_class(self.test_loader_pretrained)
        self.thresholds = self._get_thresholds_from_statistics_range(
            loss_stats=loss_stats,
            factor=self.factor_thr,
            metric=self.metric_threshould
        )
        self.model.classification = False
        print("Pretraining with classification completed.")

    def pretrain_model_contrastive(self):
        """
        Pretrains the model with contrastive learning on the specified pretrained classes.
        """
        print("Starting pretraining with contrastive learning...")
        self.model.return_encoded = True
        trainer = ContrastTrainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            alpha=self.alpha,
            batch_size=self.batch_size,
            writer=self.writer
        )
        trainer.train(
            train_loader=self.train_loader_pretrained,
            test_loader=self.test_loader_pretrained,
            epochs=self.epochs
        )
        self.training_losses.extend(trainer.training_losses)
        self.test_losses.extend(trainer.test_losses)

        self.model.return_encoded = False
        loss_stats = self._compute_losses_per_layer_per_class(self.test_loader_pretrained)
        self.thresholds = self._get_thresholds_from_statistics_range(
            loss_stats=loss_stats,
            factor=self.factor_thr,
            metric=self.metric_threshould
        )
        print("Pretraining with contrastive learning completed.")

    def apply_neurogenesis(self):
        """
        Applies the neurogenesis algorithm to incorporate new classes.
        """
        print("Starting neurogenesis...")
        self._neurogenesis()
        print("Neurogenesis completed.")
        return self.neruons_per_new_class

    def _compute_losses_per_layer_per_class(self, data_loader):
        """
        Computes reconstruction loss statistics per layer and per class, including overall statistics.
        Returns a nested dictionary: {layer_idx: {class_label: {'mean': ..., 'min': ..., 'max': ..., 'std': ..., 'median': ...}, 'overall': {...}}}
        """
        self.model.eval()
        num_layers = len(self.model.encoder_layers)
        layer_losses = {layer_idx: {} for layer_idx in range(num_layers)}  # {layer_idx: {class_label: [losses]}}
        classification = model.classification
        model.classification = False
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.view(data.size(0), -1)
                labels = labels.numpy()
                for layer_idx in range(num_layers):
                    outputs = self.model(data, l=layer_idx)
                    losses = ((outputs - data) ** 2).mean(dim=1).cpu().numpy()
                    for loss, label in zip(losses, labels):
                        if label not in layer_losses[layer_idx]:
                            layer_losses[layer_idx][label] = []
                        layer_losses[layer_idx][label].append(loss)
                        # Collect all losses under 'overall'
                        if 'overall' not in layer_losses[layer_idx]:
                            layer_losses[layer_idx]['overall'] = []
                        layer_losses[layer_idx]['overall'].append(loss)

        # Compute statistics
        statistics = {}
        for layer_idx in range(num_layers):
            statistics[layer_idx] = {}
            for class_label, losses in layer_losses[layer_idx].items():
                losses = np.array(losses)
                stats = {
                    'min': np.min(losses),
                    'max': np.max(losses),
                    'mean': np.mean(losses),
                    'median': np.median(losses),
                    'std': np.std(losses)
                }
                statistics[layer_idx][class_label] = stats
        self.pretraining_stats = statistics
        model.classification = classification
        return statistics

    def _get_thresholds_from_statistics(self, loss_stats=None, class_label='overall', metric='max', factor=1.0):
        """
        Generates a list of threshold values per layer based on the specified metric and class.

        Args:
            loss_stats: The nested dictionary containing loss statistics per layer and class.
            class_label: The class label for which to retrieve the metric ('overall' for all classes).
            metric: The metric to use ('mean', 'min', 'max', 'median', 'std').
            factor: A multiplier to apply to the metric value for safety margin.

        Returns:
            thresholds: A list of threshold values per layer.
        """
        loss_stats = loss_stats or self.pretraining_stats

        thresholds = []
        for layer_idx in range(len(self.model.encoder_layers)):
            class_stats = loss_stats[layer_idx]
            if class_label in class_stats:
                value = class_stats[class_label][metric]
                thresholds.append(value * factor)
            else:
                raise ValueError(f"Class '{class_label}' not found in statistics for layer {layer_idx}.")
        return thresholds

    def _get_thresholds_from_statistics_range(self, loss_stats=None, class_label='overall', metric='max', factor=1.0):
        """
        Generates a list of threshold values per layer based on the specified metric and class.

        Args:
            loss_stats: The nested dictionary containing loss statistics per layer and class.
            class_label: The class label for which to retrieve the metric ('overall' for all classes).
            metric: The metric to use ('mean', 'min', 'max', 'median', 'std').
            factor: A multiplier to apply to the metric value for safety margin.

        Returns:
            thresholds: A list of threshold values per layer.
        """
        loss_stats = self.pretraining_stats
        start = loss_stats[0][class_label][metric]
        print(loss_stats)
        end = factor * loss_stats[len(self.model.encoder_layers)-1][class_label][metric]
        thresholds = np.linspace(start, end, len(self.model.encoder_layers))
        return thresholds

    def _mean_samples_per_class(self, dataset: Dataset) -> float:
        """
        Calculates the mean number of samples per class in a PyTorch classification dataset.

        Parameters:
        - dataset (torch.utils.data.Dataset): The classification dataset containing samples and labels.

        Returns:
        - float: The mean number of samples per class.

        Raises:
        - ValueError: If the dataset contains no samples or no classes.
        """
        # Attempt to retrieve labels from common attributes
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            # If labels are not stored in attributes, iterate through the dataset
            labels = []
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                labels.append(label)
            labels = torch.tensor(labels)

        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        # Count the number of samples per class
        class_counts = Counter(labels)

        if not class_counts:
            raise ValueError("The dataset contains no classes.")

        # Calculate the mean number of samples per class
        mean_count = sum(class_counts.values()) / len(class_counts)
        return mean_count

    def _neurogenesis(self):
        """
        Internal method implementing the neurogenesis algorithm.
        """
        self.threshoulds = self._get_thresholds_from_statistics(factor=self.factor_thr, metric=self.metric_threshould)
        print(f'Thresholds: {self.thresholds}')
        n_samples = 5
        self.model.initial_layer_sizes = self.model.get_layer_sizes()

        for _class, class_dataset in zip(self.new_classes, self.train_datasets_new):

            print('current class:', _class)
            # Create data loaders
            train_data_new_loader, stability_loader, _stabilty_dataset = self._create_data_loaders(class_dataset)

            itt_overall = 0

            for level in range(len(self.model.encoder_layers)):
                nodes_added = 0
                outliers = self._compute_outliers(train_data_new_loader, level)

                while outliers is not None and outliers.size(0) > self.max_outliers and nodes_added < self.max_nodes[level]:
                    num_new_nodes = min(self.max_nodes[level] - nodes_added, math.ceil(outliers.size(0) * self.factor_ng))
                    self.model.add_nodes(level, num_new_nodes)
                    nodes_added += num_new_nodes

                    new_params = {'encoder': self.model.get_new_params('encoder'), 'decoder': self.model.get_new_params('decoder')}
                    old_params = {'encoder': self.model.get_old_params('encoder'), 'decoder': self.model.get_old_params('decoder')}
                    optimizer_plasticity = self._adjust_plasticity_optimizer(new_params, old_params)
                    optimizer_stability = self._adjust_stability_optimizer(new_params, old_params)

                    self.model.freeze_layers(self.model.encoder_layers[:level] + self.model.decoder_layers[-(level + 1):])

                    for e in range(self.epochs_per_iteration):
                        itt_overall += 1
                        outliers_loader = DataLoader(outliers, batch_size=self.batch_size, shuffle=True)
                        self._plasticity_phase(level, optimizer_plasticity, outliers_loader)
                        losses_per_layer = self._compute_losses_for_new_class(class_dataset)
                        self.losses_per_layer_history.append(losses_per_layer)
                        clear_output(wait=True)
                        # self._plot_layer_sizes_and_losses(title=f'After Plasticity Phase {e + 1} - Level {level} - Class {_class}')
                        self.log_custom_metrics(itt_overall)
                        self._stability_phase(level, optimizer_stability, stability_loader)
                        losses_per_layer = self._compute_losses_for_new_class(class_dataset)
                        self.losses_per_layer_history.append(losses_per_layer)
                        clear_output(wait=True)
                        # self._plot_layer_sizes_and_losses(title=f'After Stability Phase {e + 1} - Level {level} - Class {_class}')
                        self.log_custom_metrics(itt_overall)
                    self.model.unfreeze_layers(self.model.encoder_layers + self.model.decoder_layers)
                    outliers = self._compute_outliers(train_data_new_loader, level)
                    
                    for c in self.pretrained_classes:
                        # Get the images
                        original_imgs, reconstructed_imgs, synthetic_imgs = self.get_samples_and_reconstructions(c, n_samples)
                        # Log them to TensorBoard
                        self.log_samples_to_tensorboard(original_imgs, reconstructed_imgs, synthetic_imgs, c, itt_overall)


                    # Get the images
                    original_imgs, reconstructed_imgs, synthetic_imgs = self.get_samples_and_reconstructions(_class, n_samples)
                    # Log them to TensorBoard
                    self.log_samples_to_tensorboard(original_imgs, reconstructed_imgs, synthetic_imgs, _class, itt_overall)


                # Propagate changes to next layer if new nodes were added
                if nodes_added > 0 and level < len(self.model.encoder_layers) - 1:
                    self.model.freeze_layers(self.model.encoder_layers[:level + 1] + self.model.decoder_layers[-(level + 2):])
                    params_layer_L_plus_1 = list(self.model.encoder_layers[level + 1].parameters()) + \
                                            list(self.model.decoder_layers[-(level + 2)].parameters())
                    optimizer_full = optim.Adam(params_layer_L_plus_1, lr=self.lr, weight_decay=1e-2)

                    for e in range(self.epochs_per_iteration):
                        print(f'here is the epoch e: {e}')
                        itt_overall += 1

                        self._plasticity_phase(level + 1, optimizer_full, train_data_new_loader)
                        losses_per_layer = self._compute_losses_for_new_class(class_dataset)
                        self.losses_per_layer_history.append(losses_per_layer)
                        clear_output(wait=True)
                        # self._plot_layer_sizes_and_losses(title=f'After Plasticity Phase {e + 1} - Level {level + 1} next level - Class {_class}')
                        # print('new log incomming...')
                        self.log_custom_metrics(itt_overall)
                        itt_overall += 0.5

                        self._stability_phase(level + 1, optimizer_full, stability_loader)
                        losses_per_layer = self._compute_losses_for_new_class(class_dataset)
                        self.losses_per_layer_history.append(losses_per_layer)
                        clear_output(wait=True)
                        # self._plot_layer_sizes_and_losses(title=f'After Stability Phase {e + 1} - Level {level + 1} next level - Class {_class}')
                        # print('new log incomming...')
                        self.log_custom_metrics(itt_overall)
                    self.model.unfreeze_layers(self.model.encoder_layers + self.model.decoder_layers)
                    self.neruons_per_new_class[str(_class)] = model.hidden_sizes
                    model.consolidate_new_nodes()

                    # Get the images
                    original_imgs, reconstructed_imgs, synthetic_imgs = self.get_samples_and_reconstructions(_class, n_samples)

                    # Visualize them
                    # self.visualize_samples(original_imgs, reconstructed_imgs, synthetic_imgs, n_samples)

                    # Log them to TensorBoard
                    self.log_samples_to_tensorboard(original_imgs, reconstructed_imgs, synthetic_imgs, _class, itt_overall)

            self.pretrained_classes.append(_class)

        if self.losses_per_layer_history:
            self._plot_layer_sizes_and_losses(title='Final Layer Sizes and Losses After Neurogenesis')
            self.log_custom_metrics(itt_overall)

    # Additional methods within NGTrainer

    def _create_data_loaders(self, train_datasets_new=None, n=None):
        """
        Creates data loaders for the new training data and the stability dataset.

        Args:
            train_datasets_new: The dataset containing the new classes.
            batch_size: The batch size to use for the data loaders.
            model_encoder: The encoder part of the model.
            model_decoder: The decoder part of the model.
            train_loader_pretrained: DataLoader for the pretrained data.

        Returns:
            train_data_new_loader: DataLoader for the new training data.
            stability_loader: DataLoader for the stability dataset.
        """

        # Calculate mean samples per class in the new dataset
        if n == None:
            mspc = math.ceil(self._mean_samples_per_class(train_datasets_new))
        else:
            mspc = n

        # Create DataLoader for the new training data
        if not (train_datasets_new is None):
            train_data_new_loader = DataLoader(train_datasets_new, batch_size=self.batch_size, shuffle=True)
            # Generate synthetic dataset using intrinsic replay
        else:
            train_data_new_loader = None
        intrinsic_replay_dataset = create_synthetic_dataset(
            self.model.encoder, self.model.decoder, self.train_loader_pretrained, num_samples_per_class=mspc)

        # Combine the synthetic dataset with the new training data for stability´
        if not (train_datasets_new is None):
            stability_dataset = ConcatDataset([intrinsic_replay_dataset, train_datasets_new])
        else:
            stability_dataset = intrinsic_replay_dataset
        stability_loader = DataLoader(stability_dataset, batch_size=batch_size, shuffle=True)

        return train_data_new_loader, stability_loader, stability_dataset

    def _compute_outliers(self, data_loader, level):
        losses = []
        data_new_list = []
        self.model.eval()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.view(data.size(0), -1)
                output = self.model(data, l=level)
                batch_losses = ((output - data) ** 2).mean(dim=1).cpu().numpy()
                losses.extend(batch_losses)
                data_new_list.append(data)
        data_new_tensor = torch.cat(data_new_list, dim=0)
        losses = np.array(losses)
        outliers_indices = np.where(losses > self.thresholds[level])[0]
        return data_new_tensor[outliers_indices] if len(outliers_indices) > 0 else None

    def _plasticity_phase(self, level, optimizer, data_loader):
        self.model.train()
        batch_bar = tqdm(data_loader, desc=f'Plasticity Phase - Level {level}', leave=False)
        for batch in batch_bar:
            # Handle batch whether it's a tensor or a tuple (data, labels)
            if isinstance(batch, (list, tuple)):
                batch_data, _ = batch
            else:
                batch_data = batch
            batch_data = batch_data.view(batch_data.size(0), -1)  # Flatten the images
            optimizer.zero_grad()
            outputs = self.model(batch_data, l=level)
            loss = self.criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=loss.item())

    def _stability_phase(self, level, optimizer, stability_loader):
        self.model.train()
        batch_bar = tqdm(stability_loader, desc=f'Stability Phase - Level {level}', leave=False)
        for data_ir, _ in batch_bar:
            data_ir = data_ir.view(data_ir.size(0), -1)
            optimizer.zero_grad()
            outputs = self.model(data_ir, l=level)
            loss = self.criterion(outputs, data_ir)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=loss.item())

    def _adjust_plasticity_optimizer(self, new_params, old_params):
        return optim.Adam([
            # Encoder: Update new nodes only
            {'params': new_params['encoder'], 'lr': self.lr},
            {'params': old_params['encoder'], 'lr': 0},  # Freeze old encoder weights

            # Decoder: Update old weights at reduced learning rate
            {'params': old_params['decoder'], 'lr': self.lr * self.lr_factor},
            {'params': new_params['decoder'], 'lr': 0}  # Freeze new decoder weights
        ], weight_decay=1e-2)

    def _adjust_stability_optimizer(self, new_params, old_params):
        return optim.Adam([
            {'params': new_params['encoder'] + old_params['encoder'] + new_params['decoder'] + old_params['decoder'], 'lr': self.lr * self.lr_factor}
        ], weight_decay=1e-2)

    def _compute_losses_for_new_class(self, dataset):
        self.model.eval()
        losses_per_layer = []
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for l_idx in range(len(self.model.encoder_layers)):
                total_loss = 0
                num_samples = 0
                for data, _ in data_loader:
                    data = data.view(data.size(0), -1)
                    output = self.model(data, l=l_idx)
                    batch_loss = ((output - data) ** 2).sum().item()
                    total_loss += batch_loss
                    num_samples += data.size(0)
                average_loss = total_loss / num_samples
                losses_per_layer.append(average_loss)
        return losses_per_layer

    def _plot_layer_sizes_and_losses(self, title='Layer Sizes and Losses Over Time'):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Plot layer sizes
        layer_size_history = np.array(self.model.layer_size_history)
        for i in range(layer_size_history.shape[1]):
            axs[0].plot(layer_size_history[:, i], label=f'Layer {i+1}')
        axs[0].set_title('Layer Sizes Over Time')
        axs[0].set_xlabel('Neurogenesis Steps')
        axs[0].set_ylabel('Number of Nodes')
        axs[0].legend()
        axs[0].grid(True)

        # Plot losses per layer over time
        losses_per_layer_history = np.array(self.losses_per_layer_history)
        num_layers = losses_per_layer_history.shape[1]
        epochs = np.arange(len(losses_per_layer_history))
        for i in range(num_layers):
            axs[1].plot(epochs, losses_per_layer_history[:, i], label=f'Layer {i}')
        axs[1].set_title('Loss per Layer Over Time for New Class')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Average Reconstruction Loss')
        axs[1].legend()
        axs[1].grid(True)

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_metrics(self):
        """
        Plots training and test losses over epochs.
        """
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Training and Test Loss Progress')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot layer size development
        plt.subplot(1, 2, 2)
        layer_size_history = np.array(self.model.layer_size_history)
        for i in range(layer_size_history.shape[1]):
            plt.plot(layer_size_history[:, i], label=f'Layer {i+1}')
        plt.title('Layer Sizes Over Time')
        plt.xlabel('Neurogenesis Steps')
        plt.ylabel('Number of Nodes')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def log_custom_metrics(self, epoch):
        """
        Logs training and test losses over epochs and layer sizes over time to TensorBoard.
        """
        # Log training and test losses
        if hasattr(self, 'losses_per_layer_history'):
            loss_dict = {}
            for i, loss in enumerate(self.losses_per_layer_history[-1]):
                loss_dict[str(i+1)] = loss
            self.writer.add_scalars('Neurogenesis Losses', loss_dict, len(self.losses_per_layer_history))

        # Log layer sizes over time
        if hasattr(self.model, 'layer_size_history') and self.model.layer_size_history:
            # For each layer, log its size over neurogenesis steps
            layer_size_history = self.model.layer_size_history
            num_layers = len(layer_size_history[0])
            # Transpose layer_size_history to get sizes per layer over time
            transposed_layer_sizes = list(zip(*layer_size_history))
            layer_sizes_dict = {}
            for layer_idx in range(num_layers):
                layer_sizes = transposed_layer_sizes[layer_idx]
                # Log the latest size of the layer
                layer_sizes_dict[f'Layer_{layer_idx+1}'] = layer_sizes[-1]
                # # Optionally, log the entire history as scalars
                # for step, size in enumerate(layer_sizes):
                #     self.writer.add_scalar(f'Layer_{layer_idx+1}/Size', size, step)
            print(f'here are the new sizes{layer_sizes_dict}, and at epoch {epoch}')
            self.writer.add_scalars('Sizes', layer_sizes_dict, epoch)
        else:
            print("Layer size history is not available or empty.")# Training Utilities

    def get_samples_and_reconstructions(self, class_label, n, synthetic=True):
        """
        Retrieves n examples of the specified class, their reconstructions from the autoencoder,
        and n synthetic samples from intrinsic replay (if available).

        Args:
            class_label (int): The class label for which to retrieve samples.
            n (int): The number of samples to retrieve.

        Returns:
            original_images (list of Tensors): Original images from the dataset.
            reconstructed_images (list of Tensors): Reconstructed images from the autoencoder.
            synthetic_images (list of Tensors): Synthetic images from intrinsic replay (if available).
        """
        import torch
        from torch.utils.data import DataLoader, Subset

        # Initialize lists to store images
        original_images = []
        reconstructed_images = []
        synthetic_images = []

        # Retrieve n examples of the specified class from the dataset
        class_indices = [i for i, (_, label) in enumerate(self.test_dataset) if label == class_label]
        if not class_indices:
            print(f"No samples found for class {class_label} in the dataset.")
            return original_images, reconstructed_images, synthetic_images

        # Limit to n samples
        class_indices = class_indices[:n]
        subset = Subset(self.test_dataset, class_indices)
        data_loader = DataLoader(subset, batch_size=n, shuffle=False)

        # Get original images and their reconstructions
        self.model.eval()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.view(data.size(0), -1)
                output = self.model(data)
                output = output.view(-1, 1, 28, 28)  # Reshape to image format

                original_images.extend(data.view(-1, 1, 28, 28).cpu())
                reconstructed_images.extend(output.cpu())
        if synthetic:
            # Get synthetic images from intrinsic replay if available  
            _new_data_loader, _stability_loader, synthetic_dataset = self._create_data_loaders(n=n)
            # Filter synthetic dataset for the specified class
            synthetic_class_indices = [i for i, label in enumerate(synthetic_dataset.labels) if label == class_label]
            if synthetic_class_indices:
                synthetic_class_indices = synthetic_class_indices[:n]
                synthetic_subset = Subset(synthetic_dataset, synthetic_class_indices)
                synthetic_loader = DataLoader(synthetic_subset, batch_size=n, shuffle=False)

                for data, _ in synthetic_loader:
                    data = data.view(-1, 1, 28, 28)
                    synthetic_images.extend(data.cpu())
        else:
            print(f"No synthetic samples found for class {class_label} in the intrinsic replay dataset.")


        # Ensure lists are not longer than n
        original_images = original_images[:n]
        reconstructed_images = reconstructed_images[:n]
        synthetic_images = synthetic_images[:n]

        return original_images, reconstructed_images, synthetic_images

    def visualize_samples(self, original_images, reconstructed_images, synthetic_images, n):
        """
        Visualizes the original, reconstructed, and synthetic images.

        Args:
            original_images (list of Tensors): Original images.
            reconstructed_images (list of Tensors): Reconstructed images.
            synthetic_images (list of Tensors): Synthetic images.
            n (int): Number of images to display.
        """

        num_images = min(n, len(original_images))
        rows = 2 if not synthetic_images else 3
        fig, axes = plt.subplots(rows, num_images, figsize=(num_images * 2, rows * 2))

        for i in range(num_images):
            # Original images
            axes[0, i].imshow(original_images[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')

            # Reconstructed images
            axes[1, i].imshow(reconstructed_images[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')

            # Synthetic images (if available)
            if synthetic_images:
                axes[2, i].imshow(synthetic_images[i].squeeze(), cmap='gray')
                axes[2, i].axis('off')
                if i == 0:
                    axes[2, i].set_title('Intrinsic Replay')

        plt.tight_layout()
        plt.show()

    def log_samples_to_tensorboard(self, original_images, reconstructed_images, synthetic_images, class_label, step):
        """
        Logs images to TensorBoard for visualization.

        Args:
            original_images (list of Tensors): Original images.
            reconstructed_images (list of Tensors): Reconstructed images.
            synthetic_images (list of Tensors): Synthetic images.
            class_label (int): The class label.
            step (int): The current step or epoch number.
        """

        # Prepare images for grid
        original_grid = torchvision.utils.make_grid(original_images, nrow=5, normalize=True)
        reconstructed_grid = torchvision.utils.make_grid(reconstructed_images, nrow=5, normalize=True)

        self.writer.add_image(f'Class_{class_label}/Original', original_grid, global_step=step)
        self.writer.add_image(f'Class_{class_label}/Reconstructed', reconstructed_grid, global_step=step)

        if synthetic_images:
            synthetic_grid = torchvision.utils.make_grid(synthetic_images, nrow=5, normalize=True)
            self.writer.add_image(f'Class_{class_label}/Intrinsic_Replay', synthetic_grid, global_step=step)


# %%

#%%
# Main Execution Script
if __name__ == "__main__":

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Time Stamp: {time_stamp}')
    writer = SummaryWriter(log_dir=f'runs/neurogenesis_experiment_{time_stamp}')


    # Data transformations
    transform = transforms.Compose([
        #transforms.RandomRotation(degrees=(-180, 180)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Specify classes for pretraining and neurogenesis
    pretrained_classes = [2,7]
    new_classes = [0,2,3,4,5,6,8,9]
    batch_size = 64
    #act_func=nn.Sigmoid()
    act_func=nn.LeakyReLU()

    # Initialize the model [200, 200, 75, 20]
    model = NGAutoencoder(input_dim=28*28, hidden_sizes=[200, 200, 75, 20], act_func=act_func, num_classes_pretraining=len(pretrained_classes), dropout=0.5)

    # Initialize the NGTrainer
    ng_trainer = NGTrainer(model=model,
                           train_dataset=full_train_dataset,
                           test_dataset=full_test_dataset,
                           writer=writer,
                           pretrained_classes=pretrained_classes,
                           new_classes=new_classes,
                           batch_size=batch_size,
                           lr=1e-3,
                           lr_factor=1e-2,
                           epochs=11,
                           epochs_per_iteration=4,
                           max_nodes=[1500, 800, 500, 200],
                           max_outliers=5,
                           factor_thr=1,
                           factor_ng=1e-2,
                           metric_threshould='max',
                           alpha = 1
    )

    # Pretrain the model
    ng_trainer.pretrain_model()

    # Apply neurogenesis
    layergroth = ng_trainer.apply_neurogenesis()

    # Plot metrics
    ng_trainer.plot_metrics()

    print(layergroth)
# %%

