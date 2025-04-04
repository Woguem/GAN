"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a GAN model to generate image

"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

start_time = datetime.now()  # Start timer

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
z_dim = 100
batch_size = 128
lr = 0.0002
num_epochs = 20
img_size = 28
img_channels = 1

# Data processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, img_channels, img_size, img_size)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        flattened = img.view(-1, img_size * img_size)
        return self.model(flattened)

# Initialisation
G = Generator().to(device)
D = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# VisualiZation
def save_generated_images(epoch):
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        samples = G(z).cpu()
        fig, axes = plt.subplots(4, 4, figsize=(8,8))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i].permute(1,2,0).numpy()*0.5+0.5, cmap='gray')
            ax.axis('off')
        plt.savefig(f'gan_samples_epoch_{epoch}.png')
        plt.close()

# Training
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        
        # Real and fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Discriminator training
        optimizer_D.zero_grad()
        
        # Loss on real images
        D_real = D(real_imgs)
        loss_D_real = criterion(D_real, real_labels)
        
        # Loss on fake images 
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z).detach()
        D_fake = D(fake_imgs)
        loss_D_fake = criterion(D_fake, fake_labels)
        
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        
        # Generator training
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z)
        D_fake = D(fake_imgs)
        loss_G = criterion(D_fake, real_labels)  # We want to trick D
        
        loss_G.backward()
        optimizer_G.step()
        
    # Display et save
    print(f"Epoch [{epoch}/{num_epochs}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
    if epoch % 3 == 0:
        save_generated_images(epoch)

# Save models
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')




end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")


