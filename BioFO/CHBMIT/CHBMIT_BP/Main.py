import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

print('CHBMIT_BP')
length_each_layer = 2000
learning_rate = 0.0001
print('length_each_layer',length_each_layer)
#print('learning_rate',learning_rate)

class Net(nn.Module):
    def __init__(self,layer_number,length_each_layer):
        super(Net, self).__init__()
        self.conv1 = nn.Linear(1024,length_each_layer)
        self.conv2 = nn.Linear(length_each_layer,length_each_layer)
        self.conv3 = nn.Linear(length_each_layer,length_each_layer)
        self.conv4 = nn.Linear(length_each_layer,length_each_layer)
        self.conv5 = nn.Linear(length_each_layer,length_each_layer)
        self.conv6 = nn.Linear(length_each_layer,length_each_layer)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(length_each_layer, 2)
        self.layer_number = layer_number+1

    def forward(self, x):
        if self.layer_number > 0:
            x = self.conv1(x)
            #x= self.dropout1(F.relu(x))
            x = F.relu(x)
        if self.layer_number >1:
            x = self.conv2(x)
            #x = self.dropout1(F.relu(x))
            x = F.relu(x)
        if self.layer_number >2:
            x = self.conv3(x)
            #x = self.dropout3(F.relu(x))
            x = F.relu(x)
        if self.layer_number>3:
            x = self.conv4(x)
            #x = self.dropout4(F.relu(x))
            x = F.relu(x)
        if self.layer_number >4:
            x = self.conv5(x)
            #x = self.dropout4(F.relu(x))
            x = F.relu(x)
        if self.layer_number >5:
            x = self.conv6(x)
            #x = self.dropout4(F.relu(x))
            x = F.relu(x)


        x = self.fc1(x)

        output = x

        return output

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
#train_loader, test_loader = CIFAR10_loaders()

# create a validation set
for layer_number in range(6):
    model=Net(layer_number,length_each_layer)
    #print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#,weight_decay=1e-1)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(images.shape[0],-1)
            labels = labels

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
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
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
