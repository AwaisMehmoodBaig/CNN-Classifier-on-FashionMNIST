import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.cnn_model import CNNModel
import matplotlib.pyplot as plt

# Load trained model
model = CNNModel()
model.load_state_dict(torch.load("fashion_cnn.pth"))
model.eval()

# Load test dataset
transform = transforms.ToTensor()
test_dataset = datasets.FashionMNIST(root='data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# Class labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Get one batch of test images
images, labels = next(iter(test_loader))
outputs = model(images)
predicted = torch.argmax(outputs, dim=1)

# Plot the images with predictions
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i][0], cmap="gray")
    plt.title(f"P: {class_names[predicted[i]]}\nT: {class_names[labels[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()  # Show the window with the predicted images
