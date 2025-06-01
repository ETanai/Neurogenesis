import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def get_class_indices(dataset, classes=[1, 7]):
    indices = []
    for idx, target in enumerate(dataset.targets):
        if target in classes:
            indices.append(idx)
    return indices

class MNISTSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, classes=[1, 7]):
        self.dataset = dataset
        self.indices = indices
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data, target = self.dataset[self.indices[idx]]
        target = self.class_to_idx[int(target)]
        return data, target

class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[200, 200, 75, 20], num_classes=2):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        out = self.model(x)
        return out

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    num_classes = len(class_names)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Use white or black text based on the background color
    threshold = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm.shape):
        color = "white" if cm_normalized[i, j] > threshold else "black"
        plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                 horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def main():
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset (only classes 1 and 7)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_indices = get_class_indices(train_dataset_full, classes=[1, 7])
    test_indices = get_class_indices(test_dataset_full, classes=[1, 7])

    train_dataset = MNISTSubset(train_dataset_full, train_indices, classes=[1, 7])
    test_dataset = MNISTSubset(test_dataset_full, test_indices, classes=[1, 7])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, and Optimizer
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard SummaryWriter (ensure TensorBoard server is running)
    writer = SummaryWriter(log_dir='runs/mnist_mlp')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # Evaluate on test set
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        cm = confusion_matrix(all_targets, all_preds)
        cm_fig = plot_confusion_matrix(cm, class_names=['1', '7'])
        writer.add_figure('Confusion_matrix', cm_fig, global_step=epoch)
    
    writer.close()

if __name__ == '__main__':
    main()