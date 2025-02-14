import torch
from torchvision.datasets import MNIST,ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize,RandomRotation,RandomHorizontalFlip,CenterCrop
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import datetime


learning_rate = 0.0001
print('learning_rate',learning_rate)

start_time = datetime.datetime.now()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")

# load data
def Imagenet_loaders(train_batch_size=100, test_batch_size=100):

    transform=Compose([
            RandomRotation(10),      # rotate +/- 10 degrees
            RandomHorizontalFlip(),  # reverse 50% of images
            Resize(84),             # resize shortest side to 224 pixels
            CenterCrop(84),         # crop longest side to 224 pixels at center
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root='./miniimagenet', transform=transform)
    n_data = len(dataset)

    n_train = int(0.85 * n_data)
    n_valid = 0#int(0.15 * n_data)
    n_test = n_data - n_train - n_valid
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid, n_test])


    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    return train_loader, test_loader,


train_loader, test_loader = Imagenet_loaders()

# train data
inputs, targets = next(iter(train_loader))
#inputs, targets = inputs.cuda(), targets.cuda()


model = vgg16(pretrained=True)
model.classifier[-1] = nn.Linear(4096,100)
model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(100):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, 100, i+1, total_step, loss.item()))

# Test the model
# In the test phase, don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

end_time = datetime.datetime.now()
total_time = (end_time-start_time).total_seconds()
print('total time: ' + str(total_time))