import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import numpy as np
import torch.nn as nn

import os

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
print('CHBMIT_our')
layers = 6
length = 2000
input_size =1024
class_num = 2
learning_rate = 0.0001
weight_decay_number = 1e-2
print('no pow')
print('learning_rate',learning_rate)
#print('weight decay',weight_decay_number)

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
import chbmit_dataset.load_data as chb

x_train= np.zeros((1,32,32))
x_test= np.zeros((1,32,32))
y_train = np.zeros(1)
y_test = np.zeros(1)

chb_list = range(1,25,1)
acc_total = 0
#print("-------------begin-------------------")
for i in chb_list:
    #print("process for patient "+str(i))
    if i== 6 or i==14 or i==16: # we do not consider patient 6/14/16
        continue
    if i <10:
        which_patients = 'chb0'+str(i)
    else:
        which_patients = 'chb'+str(i)

    X_train_example,Y_train_example,X_val_example,Y_val_example,X_test_example,Y_test_example = chb.load_data(which_patients)
    #X_train_example = np.power(X_train_example[:,::2,:],2)
    #X_val_example = np.power(X_val_example[:,::2,:],2)
    #X_test_example = np.power(X_test_example[:,::2,:],2)

    X_train_example = X_train_example[:,::2,:]
    X_val_example = X_val_example[:,::2,:]
    X_test_example = X_test_example[:,::2,:]

    x_train=np.vstack((x_train,X_train_example.reshape(X_train_example.shape[0],32,32)))
    x_train=np.vstack((x_train,X_val_example.reshape(X_val_example.shape[0],32,32)))
    x_test=np.vstack((x_test,X_test_example.reshape(X_test_example.shape[0],32,32)))

    y_train = np.hstack((y_train,Y_train_example))
    y_train = np.hstack((y_train,Y_val_example))
    y_test = np.hstack((y_test,Y_test_example))

x_train = x_train[1:,:,:]
x_test = x_test[1:,:,:]
y_train = y_train[1:]
y_test = y_test[1:]

train_data = torch.utils.data.TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).long())
test_data = torch.utils.data.TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).long())

train_loader = torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)


#fixed_weights= torch.randn((class_num,2000))
matrix = np.zeros((2,2000))
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

    #new_model.lastfc.weight.data = torch.randn((class_num,2000))
    #new_model.lastfc.bias.data = torch.zeros((class_num))

    #print('fixed weight and zero bias')
    #print('kaiming weight and zero bias')
    #print('kaiming weight and randn bias')
    #print('kaiming normal weight and zero bias')
    print('random projection')
    new_model.lastfc.weight.data = fixed_weights

    new_model.lastfc.bias.data = torch.zeros((class_num))
    #new_model.lastfc.bias.data = torch.randn((class_num))
    #torch.nn.init.kaiming_uniform_(new_model.lastfc.weight)
    #torch.nn.init.kaiming_normal_(new_model.lastfc.weight)


    # print(new_model.lastfc.weight.data)
    #print(new_model.lastfc.bias.data)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)#,weight_decay=weight_decay_number)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(images.shape[0],-1)
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
            images = images.reshape(images.shape[0],-1)
            labels = labels
            outputs,_ = new_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network in {layer_index}th layer on the 10000 test images: {100 * correct / total}" )
