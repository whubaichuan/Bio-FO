import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize,RandomRotation,RandomHorizontalFlip,CenterCrop
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision.models as models
import datetime

start_time = datetime.datetime.now()


#dataset: 1-mnist,2-cifar10,3-cifar100
dataset = 2


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")


#load data

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


class Net_one(nn.Module):
    def __init__(self,vgg,class_num):
        super(Net_one, self).__init__()

        self.layer = vgg.features[0]

        self.flatten = nn.Flatten()
        self.lastfc = nn.Linear(65536, class_num) 
        self.bn  = nn.BatchNorm2d(self.layer.out_channels)

    def forward(self, x):
        
        x =self.layer(x)
        x_internal = F.relu(x)
        output =  self.lastfc(self.flatten(x_internal))
        return output,x_internal
    

class Net_more(nn.Module):
    def __init__(self,model,vgg,index,class_num,layer):
        super(Net_more, self).__init__()
        self.previous_model=model
        self.index = layer
        self.output_lenth=[65536,16384,32768,8192,16384,16384,4096,8192,8192,2048,2048,2048,512]
        self.layer = vgg.features[layer]

        self.flatten = nn.Flatten()
        self.lastfc = nn.Linear(self.output_lenth[index], class_num)
        self.bn  = nn.BatchNorm2d((self.layer.out_channels))

    def forward(self, x):
        _,x = self.previous_model(x)
        x =self.layer(x)
        if self.index in [0,5,10,12,17,19,24,26]:
            x_internal = F.relu(x)
        elif self.index in [2,7,14,21,28]:
            x_internal = F.relu(x)
            x_internal = F.max_pool2d(x_internal,2,2,)
        output = self.lastfc(self.flatten(x_internal))
        
        return output, x_internal

class classifier(nn.Module):
    def __init__(self,model,vgg,index,class_num,layer):
        super(classifier, self).__init__()
        self.previous_model=model
        self.index = index
        self.output_lenth=[4096,4096]#2508
        self.layer = vgg.classifier[layer]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.dropout = nn.Dropout(p=0.5,inplace=False)

        self.flatten = nn.Flatten()
        self.lastfc = nn.Linear(self.output_lenth[index], class_num)

    def forward(self, x):        
        _,x = self.previous_model(x)
        if self.index ==0:
            x = self.avgpool(x)
            x =self.layer(self.flatten(x))
            x_internal = F.relu(x)
            x_internal = self.dropout(x_internal)
            output = self.lastfc(x_internal)
        if self.index ==1:
            x =self.layer(x)
            x_internal = F.relu(x)
            x_internal = self.dropout(x_internal)
            output = self.lastfc(x_internal)

        return output,x_internal

vgg = vgg16(pretrained=False)
vgg.classifier[-1] = nn.Linear(4096,class_num)

for i,layer in enumerate([0,2, 5,7,10,12,14,17,19,21,24,26,28,0,3]):

    if i == 0:
        new_model=Net_one(vgg,class_num)

    elif i<13:
        for param in new_model.parameters():
            param.requires_grad = False
        new_model = Net_more(new_model,vgg,i,class_num,layer)
    else:
        continue
        # for param in new_model.parameters():
        #     param.requires_grad = False
        # new_model = classifier(new_model,vgg,i-13,class_num,layer)

    for name, param in new_model.named_parameters():
        if name =='lastfc.weight' or name=='lastfc.bias':
            param.requires_grad = False

    print(new_model)
    for name, param in new_model.named_parameters():
        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")

    new_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001)

    total_step = len(train_loader)
    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device) #.expand(-1,3, -1, -1) only for mnist
            labels = labels.to(device)

            # Forward pass
            outputs,_ = new_model(images)
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
            outputs,_ = new_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

end_time = datetime.datetime.now()
total_time = (end_time-start_time).total_seconds()
print('total time: ' + str(total_time))