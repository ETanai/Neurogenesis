import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

# -------- Autoencoder Model --------
class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        assert len(layer_sizes) >= 2, "Need at least input and one latent layer"

        # Encoder
        self.encoder_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

        # Decoder (symmetrical)
        self.decoder_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i - 1])
            for i in reversed(range(1, len(layer_sizes)))
        ])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i != len(self.decoder_layers) - 1:
                x = F.relu(x)
        return x

# -------- Training Setup --------
def train_autoencoder(layer_sizes, epochs=10, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    transform = transforms.ToTensor()
    full_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Split 80% train, 20% test
    train_size = int(0.8 * len(full_data))
    test_size = len(full_data) - train_size
    train_data, test_data = random_split(full_data, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = AutoEncoder(layer_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TensorBoard with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'./runs/mnist_autoencoder_{timestamp}'
    writer = SummaryWriter(log_dir)
    writer.add_text("run_metadata/timestamp", timestamp)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.view(imgs.size(0), -1).to(device)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        writer.add_scalar("Loss/test", avg_test_loss, epoch)

        # Log reconstructions (from test set)
        with torch.no_grad():
            sample_imgs = next(iter(test_loader))[0][:16].to(device)
            sample_imgs_flat = sample_imgs.view(sample_imgs.size(0), -1)
            recon_imgs = model(sample_imgs_flat).view(-1, 1, 28, 28)
            comparison = torch.cat([sample_imgs.cpu(), recon_imgs.cpu()])
            writer.add_images("Reconstruction", comparison, epoch)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    writer.close()

# -------- Run Training --------
if __name__ == "__main__":
    # 5-layer encoder → 784 → 256 → 128 → 64 → 32
    layer_sizes = [784, 200, 100, 50, 20]
    train_autoencoder(layer_sizes, epochs=20)