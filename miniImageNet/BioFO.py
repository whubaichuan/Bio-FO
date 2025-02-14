import torch
from torchvision.datasets import MNIST,ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize,RandomRotation,RandomHorizontalFlip,CenterCrop
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import datetime

start_time = datetime.datetime.now()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")

class_num =100

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

class Net_one(nn.Module):
    def __init__(self,vgg,class_num):
        super(Net_one, self).__init__()

        self.layer = vgg.features[0]
        self.flatten = nn.Flatten()
        self.lastfc = nn.Linear(451584, class_num) #451584 need to customized change
        self.bn  = nn.BatchNorm2d(self.layer.out_channels)
    def forward(self, x):
        
        x = self.layer(x)
        x_internal = F.relu(x)
        output =  self.lastfc(self.flatten(x_internal))
        return output,x_internal
    

class Net_more(nn.Module):
    def __init__(self,model,vgg,index,class_num,layer):
        super(Net_more, self).__init__()
        self.previous_model=model
        self.index = layer
        self.output_lenth=[451584,112896,225792,56448,112896,112896,25600,51200,51200,12800,12800,12800,2048]
        self.layer = vgg.features[layer]
        self.flatten = nn.Flatten()
        self.lastfc = nn.Linear(self.output_lenth[index], class_num)
        self.bn  = nn.BatchNorm2d((self.layer.out_channels))
        
    def forward(self, x):
        _,x = self.previous_model(x)
        x = self.layer(x) 
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
vgg.classifier[-1] = nn.Linear(4096,100)

for i,layer in enumerate([0,2, 5,7,10,12,14,17,19,21,24,26,28,0,3]):

    if i == 0:
        new_model=Net_one(vgg,class_num)

    elif i<13:
        for param in new_model.parameters():
            param.requires_grad = False
        new_model = Net_more(new_model,vgg,i,class_num,layer)
    else:
        continue
        #for param in new_model.parameters():
        #    param.requires_grad = False
        #new_model = classifier(new_model,vgg,i-13,class_num,layer)

    for name, param in new_model.named_parameters():
        if name =='lastfc.weight' or name=='lastfc.bias':
            param.requires_grad = False

    print(new_model)
    for name, param in new_model.named_parameters():
        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")


    new_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
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
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = new_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

end_time = datetime.datetime.now()
total_time = (end_time-start_time).total_seconds()
print('total time: ' + str(total_time))