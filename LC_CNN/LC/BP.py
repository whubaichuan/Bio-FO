import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import datetime

start_time = datetime.datetime.now()
#dataset: 1-mnist,2-cifar10,3-cifar100
dataset = 2
model_type = 1
print('dataset: '+str(dataset))
print('model_type: '+str(model_type))
epcoh_number=200
layer_channels = [16,32,64] 
print(layer_channels)
num_layers = len(layer_channels)
kernel_size = 3
stride = 1
bias = True  # Set to True if you want bias in the layers

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")


def MNIST_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([
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

    transform = Compose([
                         transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    #trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)
    #val_loader = torch.utils.data.DataLoader(valset, batch_size=train_batch_size, shuffle=False)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)
    
    

    return train_loader, test_loader#,val_loader

def CIFAR100_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)

    testset = torchvision.datasets.CIFAR100(root='/./data', train=False,
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

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            #nn.init.kaiming_uniform_(torch.empty(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2), mode='fan_in', nonlinearity='relu')
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.pad = pad=(1,1,
                        1,1)
        
        self.dropout = nn.Dropout2d(p=0.2)
        self.bn  = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = F.pad(x, self.pad, mode='constant', value=0)
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
       
        return self.dropout(F.relu(out))
        #return F.relu(out)


class LocallyConnectedNetwork(nn.Module):
    def __init__(self, in_channels, num_layers, layer_channels, output_size, kernel_size, stride, class_num,bias=False):
        super(LocallyConnectedNetwork, self).__init__()
        self.num_layers = num_layers
        # List to hold the locally connected layers
        self.locally_connected_layers = nn.ModuleList()
        
        # Create multiple LocallyConnected2d layers
        for i in range(num_layers):
            if i == 0:
                # First layer takes input channels
                layer = LocallyConnected2d(in_channels, layer_channels[i], output_size, kernel_size, stride, bias=bias)
            else:
                # Subsequent layers take previous layer's output channels
                layer = LocallyConnected2d(layer_channels[i-1], layer_channels[i], output_size, kernel_size, stride, bias=bias)
            self.locally_connected_layers.append(layer)

        self.locally_connected_layers.append(nn.AdaptiveAvgPool2d(output_size=(7, 7)))
        self.locally_connected_layers.append(nn.Linear(7*7*layer_channels[-1],class_num))
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Forward pass through each locally connected layer
        for layer in self.locally_connected_layers:
            if isinstance(layer, nn.Linear):
                x = self.dropout(x.view(x.shape[0],-1))

                x = layer(x)
            else:
                x = layer(x)

        return x


if model_type==1:
    model=LocallyConnectedNetwork(in_channels, num_layers, layer_channels, output_size, kernel_size, stride, class_num,bias)

print('class_num='+str(class_num))
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
model.train()
for epoch in range(epcoh_number):
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
model.eval()
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