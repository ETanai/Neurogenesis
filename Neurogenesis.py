# %%
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

# %%


class DynamicAutoencoder(nn.Module):
    def __init__(self, input_dim, initial_hidden_sizes, lr_new, lr_old):
        super(DynamicAutoencoder, self).__init__()
        self.lr_new = lr_new
        self.lr_old = lr_old
        layers = [nn.Linear(input_dim, initial_hidden_sizes[0])]
        layers += [nn.Linear(initial_hidden_sizes[i], initial_hidden_sizes[i + 1])
                   for i in range(len(initial_hidden_sizes) - 1)]
        self.encoder = nn.Sequential(*layers)

        layers = [nn.Linear(initial_hidden_sizes[i], initial_hidden_sizes[i - 1])
                  for i in range(len(initial_hidden_sizes) - 1, 0, -1)]
        layers += [nn.Linear(initial_hidden_sizes[0], input_dim)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, l=3):
        l = l+1
        # Encode with first l layers
        encoded = x
        encoded = self.encoder[:l](x)
        decoded = self.decoder[len(model.decoder) - l:](encoded)
        return decoded

    def add_nodes(self, layer_idx, num_new_nodes):
        # Add new nodes to the specified layer in both encoder and decoder
        state_dict = self.state_dict()
        if layer_idx < len(self.encoder):
            state_dict = self._add_nodes_to_layer(
                state_dict, num_new_nodes, layer_idx, True)
        if (len(self.decoder) - (2+layer_idx)) < len(self.decoder) and (len(self.decoder) - (2+layer_idx)) >= 0:
            state_dict = self._add_nodes_to_layer(
                state_dict, num_new_nodes, len(self.decoder) - (2+layer_idx), False)
            # Adjust the input size of the next layer if it exists
        if layer_idx + 1 < len(self.encoder):
            state_dict = self._adjust_input_size(
                state_dict, num_new_nodes, layer_idx+1, True)
        if len(self.decoder) - (1+layer_idx) < len(self.decoder) and (len(self.decoder) - (1+layer_idx)) >= 0:
            state_dict = self._adjust_input_size(
                state_dict, num_new_nodes, len(self.decoder) - (1+layer_idx), False)
        # self.load_state_dict(state_dict)
        return state_dict

    def _add_nodes_to_layer(self, state_dict, num_new_nodes, layer_idx, is_encoder):
        if is_encoder:
            coder = 'encoder'
        else:
            coder = 'decoder'
        old_weight = state_dict[f'{coder}.{layer_idx}.weight']
        old_bias = state_dict[f'{coder}.{layer_idx}.bias']

        output_dim, input_dim = old_weight.size()
        new_output_dim = output_dim + num_new_nodes

        new_weight = torch.randn(new_output_dim, input_dim)
        new_bias = torch.randn(new_output_dim)

        new_weight[:output_dim, :] = old_weight
        new_bias[:output_dim] = old_bias

        layer_new = nn.Linear(in_features=input_dim,
                              out_features=new_output_dim, bias=True)
        state_dict[f'{coder}.{layer_idx}.weight'] = new_weight
        state_dict[f'{coder}.{layer_idx}.bias'] = new_bias
        return state_dict

    def _adjust_input_size(self, state_dict, num_new_nodes, layer_idx, is_encoder):
        if is_encoder:
            coder = 'encoder'
        else:
            coder = 'decoder'
        old_weight = state_dict[f'{coder}.{layer_idx}.weight']
        output_dim, input_dim = old_weight.size()
        new_input_dim = input_dim + num_new_nodes

        new_weight = torch.randn(output_dim, new_input_dim)
        new_weight[:, :input_dim] = old_weight

        next_layer_new = nn.Linear(in_features=new_input_dim,
                                   out_features=output_dim, bias=True)

        state_dict[f'{coder}.{layer_idx}.weight'] = new_weight
        return state_dict

    def get_new_params(self, encoder_size, decoder_size):
        new_params = []

        # Check new parameters in the encoder
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                original_output_dim = encoder_size[i][1]
                num_new_nodes = layer.weight.size(1) - original_output_dim
                if num_new_nodes > 0:
                    new_params.append(
                        {'params': layer.weight[:, -num_new_nodes:], 'lr': self.lr_new})
                    new_params.append(
                        {'params': layer.bias[-num_new_nodes:], 'lr': self.lr_new})

        # Check new parameters in the decoder
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Linear):
                original_output_dim = decoder_size[i][1]
                num_new_nodes = layer.weight.size(1) - original_output_dim
                if num_new_nodes > 0:
                    new_params.append(
                        {'params': layer.weight[:, -num_new_nodes:], 'lr': self.lr_new})
                    new_params.append(
                        {'params': layer.bias[-num_new_nodes:], 'lr': self.lr_new})

        return new_params


def train_and_test(model, train_loader, test_loader, criterion, optimizer, epochs=1000):
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    training_losses = []
    test_losses = []
    epoch = 0

    epoch_bar = tqdm(range(epochs), desc='Total Progress', leave=True)
    for epoch in epoch_bar:
        batch_losses = []
        # Use inner display for batch-level progress
        batch_bar = tqdm(
            train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for data, _ in batch_bar:
            data = data.view(data.size(0), -1)  # Flatten the images
            output = model(data)
            loss = criterion(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
        scheduler.step()

        training_loss = sum(batch_losses) / len(batch_losses)
        training_losses.append(training_loss)

        # Calculate test loss after each epoch
        test_loss = test(model, test_loader, criterion)
        test_losses.append(test_loss)

        # Update only the plot after all batches in an epoch are processed
        live_plot(training_losses, test_losses,
                  title=f'Training and Test Loss Progress: Epoch {epoch + 1}')

# Function to calculate test loss


def test(model, test_loader, criterion, l=3, complete=False):
    model.eval()
    total_loss = 0
    all_losses = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1)
            output = model(data, l)
            loss = criterion(output, data)
            total_loss += loss.item()
            all_losses.append(loss)
    if complete:
        ret = all_losses
    else:
        ret = total_loss/len(test_loader)
    return ret


# %%
def live_plot(training_losses, test_losses, figsize=(7, 5), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def create_optimizer(model, base_lr, new_lr):

    base_params = []
    new_params = []
    for name, param in model.named_parameters():
        # Check if this parameter is among the new neurons
        is_new = any(name.startswith(f'encoder.{idx}.') and idx_slice.start <= idx <= idx_slice.stop
                     for idx, idx_slice in model.new_neurons['encoder'])
        if is_new:
            new_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': base_lr},
        {'params': new_params, 'lr': new_lr}
    ])
    return optimizer


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)


def filter_by_class(dataset, classes):
    indices = [i for i, (img, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)


def get_class_stats(encoder, data_loader):
    encoder.eval()  # Set the model to evaluation mode
    class_predictions = {}  # Initialize an empty dictionary for storing predictions

    with torch.no_grad():  # No need to track gradients
        for images, labels in data_loader:
            images = images.view(images.shape[0], -1)  # Flatten the images
            outputs = encoder(images)
            if not isinstance(labels, list):
                labels = [labels]

            for label, probability in zip(labels, outputs):
                if not isinstance(label, int):
                    label = label.item()
                if label not in class_predictions:
                    class_predictions[label] = []
                class_predictions[label].append(
                    probability.numpy())  # Store probabilities

    # Calculate mean and standard deviation for each class
    class_stats = {}
    for label, predictions in class_predictions.items():
        # Convert list to tensor for statistical computation
        predictions = torch.tensor(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        class_stats[label] = {'mean': mean, 'std': std}

    return class_stats


def get_class_stats_with_cholesky(encoder, data_loader):
    encoder.eval()  # Set the model to evaluation mode
    class_predictions = {}  # Initialize an empty dictionary for storing predictions

    with torch.no_grad():  # No need to track gradients
        for images, labels in data_loader:
            images = images.view(images.shape[0], -1)  # Flatten the images
            outputs = encoder(images)
            if not isinstance(labels, list):
                labels = [labels]

            for label, probability in zip(labels, outputs):
                if not isinstance(label, int):
                    label = label.item()
                if label not in class_predictions:
                    class_predictions[label] = []
                class_predictions[label].append(
                    probability.numpy())  # Store probabilities

    # Calculate mean and Cholesky factorization of covariance for each class
    class_stats = {}
    for label, predictions in class_predictions.items():
        # Convert list to tensor for statistical computation
        predictions = torch.tensor(predictions)
        mean = predictions.mean(dim=0)
        # Calculate the covariance matrix
        # Transpose for correct dimensionality
        cov_matrix = torch.cov(predictions.t())
        # Cholesky decomposition of the covariance matrix
        chol_cov = torch.linalg.cholesky(cov_matrix)
        class_stats[label] = {'mean': mean, 'cholesky': chol_cov}

    return class_stats


def sample_from_stats(class_stats, num_samples_per_class):
    sampled_outputs = {}

    for class_label, stats in class_stats.items():
        mean = stats['mean'].unsqueeze(0)  # Adding batch dimension
        std = stats['std'].unsqueeze(0)  # Adding batch dimension

        # Replicate mean and std for the number of samples
        mean = mean.repeat(num_samples_per_class, 1)
        std = std.repeat(num_samples_per_class, 1)

        # Sampling from the Gaussian distribution
        samples = torch.normal(mean, std)
        sampled_outputs[class_label] = samples

    return sampled_outputs


def sample_from_stats(class_stats, num_samples_per_class):
    sampled_outputs = {}

    for class_label, stats in class_stats.items():
        mean = stats['mean'].unsqueeze(0)  # Adding batch dimension
        std = stats['std'].unsqueeze(0)  # Adding batch dimension

        # Replicate mean and std for the number of samples
        mean = mean.repeat(num_samples_per_class, 1)
        std = std.repeat(num_samples_per_class, 1)

        # Sampling from the Gaussian distribution
        samples = torch.normal(mean, std)
        sampled_outputs[class_label] = samples

    return sampled_outputs


class SyntheticMNIST(Dataset):
    def __init__(self, data, labels):
        self.data = data  # This should be a list of tensors (images)
        self.labels = labels  # Corresponding labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_synthetic_dataset(encoder, decoder, data_loader, num_samples_per_class):
    encoder.eval()
    decoder.eval()
    class_stats = get_class_stats(encoder, data_loader)
    sampled_data = sample_from_stats(class_stats, num_samples_per_class)

    synthetic_images = []
    synthetic_labels = []
    for label, samples in sampled_data.items():
        for sample in samples:
            # Decode sample to an image
            # Unsqueeze to add batch dimension
            image = decoder(sample.unsqueeze(0))
            image = image.view(-1, 28, 28)  # Reshape output to 28x28 image
            synthetic_images.append(image)
            synthetic_labels.append(label)

    synthetic_dataset = SyntheticMNIST(synthetic_images, synthetic_labels)
    return synthetic_dataset


def count_samples_per_class(dataset):
    class_counts = {}
    for _, label in dataset:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    return class_counts


def average_dict_values(data_dict):
    total = sum(data_dict.values())
    count = len(data_dict)
    # To handle empty dictionaries safely
    average = total / count if count > 0 else 0
    return average


def get_intrinsic_replay_dataset(model, new_classes, dataset_full, dataset_trained,  factor=1):
    dataset_new_classes = filter_by_class(dataset_full, new_classes)
    samplesize = math.ceil(average_dict_values(
        count_samples_per_class(dataset_new_classes))*factor)
    dataset_intrinsic_replay = create_synthetic_dataset(
        model.encoder, model.decoder, dataset_trained, samplesize)
    dataset_new = ConcatDataset(
        [dataset_new_classes, dataset_intrinsic_replay])
    return dataset_new


def freeze_layers(layers):
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False


def unfreeze_layers(layers):
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True


def split_parameters(model, sizes):
    old_params = []
    new_params = []

    def add_params(layer_params, size):
        old_params_layer = []
        new_params_layer = []

        weight, bias = layer_params
        input_dim, output_dim = size

        # Split the weight
        old_weight = weight[:, :output_dim]
        new_weight = weight[:, output_dim:]
        old_params_layer.append(old_weight)
        if new_weight.size(1) > 0:  # Check if there are new weights
            new_params_layer.append(new_weight)

        # Split the bias
        old_bias = bias[:output_dim]
        new_bias = bias[output_dim:]
        old_params_layer.append(old_bias)
        if new_bias.size(0) > 0:  # Check if there are new biases
            new_params_layer.append(new_bias)

        old_params.extend(old_params_layer)
        new_params.extend(new_params_layer)

    # Iterate through the layers of the model
    for i, layer in enumerate(model.encoder + model.decoder):
        if isinstance(layer, nn.Linear):
            weight = layer.weight
            bias = layer.bias
            layer_params = (weight, bias)

            if i < len(sizes):  # Compare with the original sizes if available
                size = sizes[i]
            else:
                raise ValueError(
                    "The sizes list is shorter than the number of layers in the model.")

            add_params(layer_params, size)

    return old_params, new_params


def get_hidden_sizes_from_state_dict(state_dict):
    keys = list(state_dict.keys())
    encoder_keys = [
        key for key in keys if 'encoder' in key and 'weight' in key]
    hidden_sizes = []
    for key in encoder_keys:
        in_features = state_dict[key].shape[1]
        hidden_sizes.append(in_features)
    hidden_sizes.append(state_dict[encoder_keys[-1]].shape[0])
    return hidden_sizes


def split_model_parameters(model, size_spec):
    new_params = {}
    old_params = {}
    model_state_dict = model.state_dict()
    for size_old, params in zip(size_spec, model_state_dict):
        print('size_old', size_old)
        if len(size_old) == 2:
            new_params[params] = model_state_dict[params][size_old[1]:, :]
            old_params[params] = model_state_dict[params][:size_old[1], :]
        if len(size_old) == 1:
            old_params[params] = model_state_dict[params][:size_old[0]]
            new_params[params] = model_state_dict[params][size_old[0]:]
    old_params = [old_params[p] for p in old_params]
    new_params = [new_params[p] for p in new_params]

    return old_params, new_params

# %%


classes_of_interest = [1, 7]
train_dataset = filter_by_class(full_train_dataset, classes_of_interest)
test_dataset = filter_by_class(full_test_dataset, classes_of_interest)
batchsize = 64

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
# %%
criterion = nn.MSELoss()
model = DynamicAutoencoder(784, [200, 200, 75, 20], 1e-3, 1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_and_test(model, train_loader, test_loader, criterion, optimizer, 5)
ds_ir = get_intrinsic_replay_dataset(
    model, [7], full_train_dataset, train_dataset)
classes = [2]
train_data_2 = filter_by_class(full_train_dataset, classes)
train_data_2 = DataLoader(train_data_2, batch_size=batchsize, shuffle=True)
# %%


def neurogenesis(model, data_new, data_old, threshoulds, max_nodes, max_outliers, pretrained_digits, classes, lr=1e-4, criterion=nn.MSELoss(), epochs_per_it=1,):
    ir_training_data = get_intrinsic_replay_dataset(
        model, pretrained_digits, full_train_dataset, train_dataset)
    size_start = [l.size() for l in model.parameters()]
    for level in range(len(model.decoder)):
        nodes_new = 0
        losses = test(model, data_new, criterion, level+1, True)
        indices = [i for i, value in enumerate(
            losses) if value > threshoulds[level]]
        outliers = [data_new.dataset[i][0].view(
            data_new.dataset[i][0].size(0), -1) for i in indices]
        data_outliers = torch.stack(outliers)

        while len(outliers) > max_outliers and nodes_new < max_nodes[level]:
            # Plasticety
            state_dict_new = model.add_nodes(
                level, math.ceil(len(outliers)/10))
            modeldimensions = get_hidden_sizes_from_state_dict(state_dict_new)
            model = DynamicAutoencoder(
                modeldimensions[0], modeldimensions[1:], 1e-3, 1e-4)
            model.load_state_dict(state_dict_new)
            new_nodes_lr = 0.001
            existing_nodes_lr = new_nodes_lr/100

            new_params, existing_params = split_model_parameters(
                model, size_start)

            optimizer_new = torch.optim.Adam([
                {'params': new_params, 'lr': new_nodes_lr},
                {'params': existing_params, 'lr': existing_nodes_lr}
            ])
            for epoch in range(epochs_per_it):
                for data in data_new:
                    optimizer_new.zero_grad()
                    # Use truncated model
                    outputs = model(data_outliers, level)
                    loss = criterion(outputs, data_outliers)
                    loss.backward()
                    optimizer_new.step()

            # stability
            # data_ir = create_synthetic_dataset(
            #     model.encoder, model.decoder, data_old, len(data_new))
            optimizer_old = torch.optim.Adam(
                model.parameters(), existing_nodes_lr)
            for epoch in range(epochs_per_it):
                for data, labels in ir_training_data:
                    optimizer_old.zero_grad()
                    # Use truncated model
                    outputs = model(data.view(-1), l=level)
                    loss = criterion(outputs, data.view(-1))
                    loss.backward(retain_graph=True)
                    optimizer_old.step()

            losses = test(model, data_new, criterion, level, True)
            indices = [i for i, value in enumerate(
                losses) if value > threshoulds[level]]
            outliers = [data_new.dataset[i][0] for i in indices]
            nodes_new = nodes_new + math.ceil(len(outliers)/10)

        if nodes_new > 0 and level < len(model.encoder):
            freeze_layers(model.encoder[:level])
            freeze_layers(model.decoder[-level:])
            for epoch in range(epochs_per_it):
                for data, labels in data_new:
                    optimizer_new.zero_grad()
                    # Use truncated model
                    outputs = model(data.view(-1), level+1)
                    loss = criterion(outputs, data.view(-1))
                    loss.backward()
                    optimizer_new.step()
            for epoch in range(epochs_per_it):
                for data, labels in ir_training_data:
                    optimizer_old.zero_grad()
                    # Use truncated model
                    outputs = model(data.view(-1), level+1)
                    loss = criterion(outputs, data.view(-1))
                    loss.backward()
                    optimizer_old.step()
            unfreeze_layers(model.encoder[:level])
            unfreeze_layers(model.decoder[-level:])

    return


neurogenesis(model, train_data_2, train_loader, [
             1, 0.8, 0.6, 0.4], [1500, 800, 500, 200], 5, [1, 7], classes)
# %%
