import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize,RandomRotation,RandomHorizontalFlip,CenterCrop
from torch.utils.data import DataLoader
from torchvision.models import vgg16,resnet18
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import datetime

#dataset: 1-mnist,2-cifar10,3-cifar100
dataset = 2

start_time = datetime.datetime.now()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")

# load data
def CIFAR10_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                        #Lambda(lambda x: torch.flatten(x))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def CIFAR100_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)

    return train_loader, test_loader

def MNIST_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([
        Resize(32),
        ToTensor()])
        #Normalize((0.1307,), (0.3081,)),
        #Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=False)  # True

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader



if dataset==1:
    #input_size =784
    in_channels = 1
    output_size = 28  # Desired output size after all layers
    class_num = 10
    learning_rate = 0.0001
    print('Dataset: mnist')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = MNIST_loaders()
if dataset==2:
    #input_size =3072
    in_channels = 3
    output_size = 32  # Desired output size after all layers
    class_num = 10
    learning_rate = 0.0001
    print('Dataset: cifar10')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR10_loaders()
elif dataset==3:
    #input_size =3072
    in_channels = 3
    output_size = 32  # Desired output size after all layers
    class_num = 100
    learning_rate = 0.0001
    print('Dataset: cifar100')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR100_loaders()


model = vgg16(pretrained=False)
model.classifier[-1] = nn.Linear(4096,class_num)

model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train the model
total_step = len(train_loader)
for epoch in range(100):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device) #.expand(-1,3, -1, -1) only for mnist
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
        images = images.expand(-1,3, -1, -1).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

end_time = datetime.datetime.now()
total_time = (end_time-start_time).total_seconds()
print('total time: ' + str(total_time))