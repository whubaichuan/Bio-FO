import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


import torch.nn as nn

import os

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np

layers = 6
length = 2000
input_size =3072
class_num = 100

print('CIFAR100_our')

# with random fix Matrix

class Net_one(nn.Module):
    def __init__(self,length,input_size,class_num):
        super(Net_one, self).__init__()
        self.fc = nn.Linear(input_size,length)

        self.lastfc = nn.Linear(length, class_num)

    def forward(self, x):
        x = self.fc(x)
        x_internal = F.relu(x)

        x = self.lastfc(x_internal)

        output = x

        return output,x_internal


class Net_more(nn.Module):
    def __init__(self,model,input_features, out_features,class_num):
        super(Net_more, self).__init__()
        self.previous_model=model
        self.fc = nn.Linear(input_features, out_features)
        self.lastfc = nn.Linear(out_features, class_num)

    def forward(self, x):
        _,x = self.previous_model(x)
        x_internal = F.relu(self.fc(x))
        x = self.lastfc(x_internal)
        output = x
        return output,x_internal

# load data
def CIFAR100_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        Lambda(lambda x: torch.flatten(x))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)

    return train_loader, test_loader


train_loader, test_loader = CIFAR100_loaders()

#fixed_weights = torch.randn((100,2000))
matrix = np.zeros((100,2000))
total_elements = length*class_num
one_third_elements = total_elements // 6
indices_to_set_to_one = np.random.choice(total_elements, one_third_elements, replace=False)
indices_to_set_to_negone = np.random.choice(np.delete(np.arange(total_elements),indices_to_set_to_one), one_third_elements, replace=False)
matrix.flat[indices_to_set_to_one] = 1
matrix.flat[indices_to_set_to_negone] = -1
fixed_weights = torch.tensor(matrix).to(dtype=torch.float32)


for layer_index in range(layers):
    if layer_index == 0:
        new_model=Net_one(length,input_size,class_num)

    else: 
        for param in new_model.parameters():
            param.requires_grad = False
        new_model = Net_more(new_model,length,length,class_num)
    

    for name, param in new_model.named_parameters():
        if name =='lastfc.weight' or name=='lastfc.bias':
            param.requires_grad = False

    # print(new_model)
    for name, param in new_model.named_parameters():
        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")

    # print(new_model.lastfc.weight.data)
    #print(new_model.lastfc.bias.data)

    #new_model.lastfc.weight.data = torch.randn((100,2000))
    #new_model.lastfc.bias.data = torch.zeros((100))

    #print('fixed weight and zero bias')
    #print('kaiming weight and zero bias')
    #print('kaiming weight and randn bias')
    #print('kaiming normal weight and zero bias')
    print('random projection')
    new_model.lastfc.weight.data = fixed_weights

    new_model.lastfc.bias.data = torch.zeros((100))
    #new_model.lastfc.bias.data = torch.randn((100))
    #torch.nn.init.kaiming_uniform_(new_model.lastfc.weight)
    #torch.nn.init.kaiming_normal_(new_model.lastfc.weight)
    
    # print(new_model.lastfc.weight.data)
    #print(new_model.lastfc.bias.data)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images
            labels = labels
            
            # Forward pass
            outputs,_= new_model(images)
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
            images = images
            labels = labels
            outputs,_ = new_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network in {layer_index}th layer on the 10000 test images: {100 * correct / total}" )
