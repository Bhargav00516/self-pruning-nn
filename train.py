import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import PruningNet
from utils import compute_sparsity_loss, calculate_sparsity, evaluate

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ✅ UPDATED lambda values
lambda_values = [0.01, 0.1, 1.0]

results = []

for lambda_val in lambda_values:
    print(f"\nTraining with lambda = {lambda_val}")

    model = PruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Reduced epochs for speed
    for epoch in range(3):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            classification_loss = criterion(outputs, labels)
            sparsity_loss = compute_sparsity_loss(model)

            loss = classification_loss + lambda_val * sparsity_loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed")

    accuracy = evaluate(model, testloader)
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {accuracy:.2f}% | Sparsity: {sparsity:.2f}%")

    results.append((lambda_val, accuracy, sparsity))

# Plot gate distribution
gates_all = []
for module in model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
        gates_all.extend(gates.flatten())

plt.hist(gates_all, bins=50)
plt.title("Gate Distribution")
plt.savefig("results.png")

# Print results
print("\nFinal Results:")
for r in results:
    print(f"Lambda: {r[0]} | Accuracy: {r[1]:.2f}% | Sparsity: {r[2]:.2f}%")