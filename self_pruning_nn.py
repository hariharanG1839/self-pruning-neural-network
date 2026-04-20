# ==============================
# 1. IMPORTS (TOP OF FILE)
# ==============================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os

# ==============================
# 2. CONFIG
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ==============================
# 3. LOAD CIFAR-10 DATASET
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# ==============================
# 4. TEST DATA LOADING
# ==============================
data_iter = iter(train_loader)
images, labels = next(data_iter)

print("Images shape:", images.shape)
print("Labels:", labels)

# ==============================
# 5. PRUNABLE LINEAR LAYER
# ==============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Standard weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores (same shape as weight)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Convert scores → gates (0 to 1)
        gates = torch.sigmoid(self.gate_scores)

        # Apply gates to weights
        pruned_weights = self.weight * gates

        # Linear operation
        return F.linear(x, pruned_weights, self.bias)

# ==============================
# 6. TEST PRUNABLE LINEAR
# ==============================
layer = PrunableLinear(10, 5).to(DEVICE)

# dummy input (batch_size=2, features=10)
x = torch.randn(2, 10).to(DEVICE)

output = layer(x)

print("PrunableLinear output shape:", output.shape) 

# ==============================
# 7. MODEL DEFINITION
# ==============================
class PrunableNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        # Flatten image
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# ==============================
# 8. TEST MODEL FORWARD PASS
# ==============================
model = PrunableNN().to(DEVICE)

# dummy CIFAR-like input (batch=4)
x = torch.randn(4, 3, 32, 32).to(DEVICE)

output = model(x)

print("Model output shape:", output.shape)

# ==============================
# 9. SPARSITY LOSS
# ==============================
def compute_sparsity_loss(model):
    sparsity_loss = 0

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)
            sparsity_loss += gates.mean() 

    return sparsity_loss

# ==============================
# 10. TRAINING SETUP
# ==============================
model = PrunableNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

LAMBDA = 0.1
EPOCHS = 2

# ==============================
# 11. TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        classification_loss = criterion(outputs, labels)

        sparsity_loss = compute_sparsity_loss(model)

        loss = classification_loss + LAMBDA * sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(train_loader):.4f}")

# ==============================
# 12. EVALUATION (ACCURACY)
# ==============================
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

#========================
#STEP 13: Calculate Sparcity 
#=========================

def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)

            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total

# ==============================
# PLOT & SAVE GATE DISTRIBUTION
# ==============================

def plot_gate_distribution(model, filename="gate_distribution.png"):
    """
    Plots and saves histogram of gate values.
    Saves image in the same directory as the script.
    """

    all_gates = []

    # Collect all gate values
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    # Create histogram
    plt.figure()
    plt.hist(all_gates, bins=50)

    plt.title("Distribution of Gate Values")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")

    # Save in same folder as script
    save_path = os.path.join(os.getcwd(), filename)
    plt.savefig(save_path)

    print(f"Gate distribution plot saved at: {save_path}")

    plt.close()


# ==============================
# RUN FINAL EVALUATION
# ==============================
acc = evaluate(model, test_loader)
print(f"Test Accuracy: {acc:.2f}%")

sparsity = calculate_sparsity(model)
print(f"Sparsity: {sparsity:.2f}%")

plot_gate_distribution(model)