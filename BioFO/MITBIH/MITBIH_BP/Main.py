import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

print('MITBIH_BP')

class Net(nn.Module):
    def __init__(self,layer_number):
        super(Net, self).__init__()
        self.conv1 = nn.Linear(169,2000)
        self.conv2 = nn.Linear(2000,2000)
        self.conv3 = nn.Linear(2000,2000)
        self.conv4 = nn.Linear(2000,2000)
        self.conv5 = nn.Linear(2000,2000)
        self.conv6 = nn.Linear(2000,2000)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(2000, 5)
        self.layer_number = layer_number+1

    def forward(self, x):
        if self.layer_number > 0:
            x = self.conv1(x)
            # = self.dropout1(F.relu(x))
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
import mitbih_dataset.load_data as bih
balance = 1 # 0 means no balanced (raw data), and 1 means balanced (weighted selected).
# Please see .mithib_dataset/Distribution.png for more data structure and distribution information.
# The above .png is from the paper-Zhang, Dengqing, et al. "An ECG heartbeat classification method based on deep convolutional neural network." Journal of Healthcare Engineering 2021 (2021): 1-9.
x_train, y_train, x_test, y_test = bih.load_data(balance)

x_train = x_train[:,:169,:].reshape((x_train.shape[0],13,13))
x_test = x_test[:,:169,:].reshape((x_test.shape[0],13,13))
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_data=torch.utils.data.TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).long())
test_data=torch.utils.data.TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).long())

train_loader = torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)


#train_loader, test_loader = CIFAR10_loaders()

# create a validation set
for layer_number in range(6):
    model=Net(layer_number)
    #print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
