import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#1. download data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),  #transfer to pytorch Tensor 
    transforms.Normalize((0.5,), (0.5,))
])

#MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. define CNN models
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # full connection layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # first layer conv + ReLU
        x = F.max_pool2d(x, 2)     # max pooling
        x = F.relu(self.conv2(x))  # second layer conv + ReLU
        x = F.max_pool2d(x, 2)     # max pooling
        x = x.view(-1, 64 * 7 * 7) # flat
        x = F.relu(self.fc1(x))    # full connection + ReLUs
        x = self.fc2(x)            # output
        return x

# instance
model = SimpleCNN()
model = model.to(device)

# 3. loss func and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. train
num_epochs = 5
model.train() 
print(f"Running on {device}")
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)  
        loss = criterion(outputs, labels) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# 5. test
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 6. visualize
dataiter = iter(test_loader)
images, labels = next(dataiter)
images=images.to(device)
labels=labels.to(device)
outputs = model(images)
_, predictions = torch.max(outputs, 1)

fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
    bias=3#show different images
    images=images.cpu()
    axes[i].imshow(images[i+bias][0], cmap='gray')
    axes[i].set_title(f"Label: {labels[i+bias]}\nPred: {predictions[i+bias]}")
    axes[i].axis('off')
plt.show()