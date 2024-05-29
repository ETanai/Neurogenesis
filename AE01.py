# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.optim.lr_scheduler import ExponentialLR


# %%
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Linear(100, 75),
            nn.ReLU(True),
            nn.Linear(75, 20)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(20, 75),
            nn.ReLU(True),
            nn.Linear(75, 100),
            nn.ReLU(True),
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Linear(200, 784),
            nn.Sigmoid()  # Use sigmoid to ensure the output is between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# %%


# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_train = datasets.MNIST(
    './data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Initialize model, loss and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Function to update the plot for both training and testing loss
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

# Training Loop with Nested Progress Bars and Testing Loss Calculation


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


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            loss = criterion(output, data)
            total_loss += loss.item()
    return total_loss / len(test_loader)


# Assuming model, train_loader, test_loader, criterion, optimizer are already defined
train_and_test(model, train_loader, test_loader, criterion, optimizer)
# Testing function call
test(model, test_loader)
''  # %%

# %%
