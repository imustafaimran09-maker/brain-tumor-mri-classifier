from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
 
path = "/Users/mustaphaemraan/.cache/kagglehub/datasets/tombackert/brain-tumor-mri-data/versions/1/brain-tumor-mri-dataset"


# for folder in os.listdir(path):
#     folder_path = os.path.join(path, folder)
#     if os.path.isdir(folder_path):
#         count = len(os.listdir(folder_path))
#         print(f"{folder}: {count} images")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(path, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)


model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    
    print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.3f} | Accuracy: {100*correct/total:.2f}%")
