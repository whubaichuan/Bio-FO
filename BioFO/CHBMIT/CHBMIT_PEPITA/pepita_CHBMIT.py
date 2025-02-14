import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import chbmit_dataset.load_data as chb
import torch.utils.data as Data

print('CHBMIT_PEPITA')
layers = [1024, 1024,1024,1024]
classes = 2
epoch_set = 100
length_network = len(layers)-1
print('layers: ' + str(length_network))
print(layers)

# models with Dropout

class NetFC1x1024DOcust(nn.Module):
    def __init__(self,dims,classes):
        super().__init__()
        #self.layers = []
        self.layers = torch.nn.ModuleDict(
            {f"fc{d+1}": nn.Linear(dims[d],dims[d+1],bias=False) for d in range(len(dims)-1)}
            )

        for d in range(len(dims)-1):
            #self.layers+=[nn.Linear(dims[d],dims[d+1],bias=False)]
            nin = dims[d]
            limit = np.sqrt(6.0 / nin)
            torch.nn.init.uniform_(self.layers[f"fc{d+1}"].weight, a=-limit, b=limit)

        self.fc_last = nn.Linear(dims[-1], classes,bias=False)
        fc_last_nin = dims[-1]
        fc_last_limit = np.sqrt(6.0 / fc_last_nin)
        torch.nn.init.uniform_(self.fc_last.weight, a=-fc_last_limit, b=fc_last_limit)


    def forward(self, x, do_masks):
        x = F.relu(self.layers[f"fc{1}"](x))
        # apply dropout --> we use a custom dropout implementation because we need to present the same dropout mask in the two forward passes
        if do_masks is not None:
            i = 0
            for i in range(1,len(self.layers)):
                x = x * do_masks[i-1]
                x = F.relu(self.layers[f"fc{i+1}"](x))
            x = x * do_masks[i]
            x = F.softmax(self.fc_last(x),dim=1)

        else:
            for i in range(1,len(self.layers)):
                x = F.relu(self.layers[f"fc{i+1}"](x))
            x = F.softmax(self.fc_last(x),dim=1)
        return x

# set hyperparameters
## learning rate
eta = 0.01
## dropout keep rate
keep_rate = 0.9
## loss --> used to monitor performance, but not for parameter updates (PEPITA does not backpropagate the loss)
criterion = nn.CrossEntropyLoss()
## optimizer (choose 'SGD' o 'mom')
optim = 'mom' # --> default in the paper
if optim == 'SGD':
    gamma = 0
elif optim == 'mom':
    gamma = 0.9
## batch size
batch_size = 100 # --> default in the paper


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
    # X_train_example = np.power(X_train_example[:,::2,:],2)
    # X_val_example = np.power(X_val_example[:,::2,:],2)
    # X_test_example = np.power(X_test_example[:,::2,:],2)

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


net = NetFC1x1024DOcust(layers,classes)


# define function to register the activations --> we need this to compare the activations in the two forward passes
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
for name, layer in net.named_modules():
    #print(name,'---',layer)
    layer.register_forward_hook(get_activation(name))

# define B --> this is the F projection matrix in the paper (here named B because F is torch.nn.functional)
nin = layers[0]
sd = np.sqrt(6/nin)
B = (torch.rand(nin,classes)*2*sd-sd)*0.05  # B is initialized with the He uniform initialization (like the forward weights)

# do one forward pass to get the activation size needed for setting up the dropout masks
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = torch.flatten(images, 1) # flatten all dimensions except batch
outputs = net(images,do_masks=None)
layers_act = []
for key in activation:
    if 'fc' in key or 'conv' in key:
        layers_act.append(F.relu(activation[key]))

# set up for momentum
if optim == 'mom':
    gamma = 0.9
    v_w_all = []
    for l_idx,w in enumerate(net.parameters()):
        #print(l_idx,'---',w.size())
        if len(w.shape)>1:
            with torch.no_grad():
                v_w_all.append(torch.zeros(w.shape))

# Train and test the model
test_accs = []
for epoch in range(epoch_set):  # loop over the dataset multiple times

    # learning rate decay
    if epoch in [60,90]:
        eta = eta*0.1
        print('eta decreased to ',eta)

    # loop over batches
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, target = data
        inputs = torch.flatten(inputs, 1) # flatten all dimensions except batch
        target_onehot = F.one_hot(target,num_classes=classes)

        # create dropout mask for the two forward passes --> we need to use the same mask for the two passes
        do_masks = []
        if keep_rate < 1:
            for l in layers_act[:-1]:
                input1 = l
                do_mask = Variable(torch.ones(inputs.shape[0],input1.data.new(input1.data.size()).shape[1]).bernoulli_(keep_rate))/keep_rate
                do_masks.append(do_mask)
            do_masks.append(1) # for the last layer we don't use dropout --> just set a scalar 1 (needed for when we register activation layer)

        # forward pass 1 with original input --> keep track of activations
        outputs = net(inputs,do_masks)
        layers_act = []
        cnt_act = 0
        for key in activation:
            if 'fc' in key or 'conv' in key:
                layers_act.append(F.relu(activation[key])* do_masks[cnt_act]) # Note: we need to register the activations taking into account non-linearity and dropout mask
                #layers_act.append(F.relu(activation[key]))
                cnt_act += 1

        # compute the error
        error = outputs - target_onehot

        # modify the input with the error
        error_input = error @ B.T
        mod_inputs = inputs + error_input

        # forward pass 2 with modified input --> keep track of modulated activations
        mod_outputs = net(mod_inputs,do_masks)
        mod_layers_act = []
        cnt_act = 0
        for key in activation:
            if 'fc' in key or 'conv' in key:
                mod_layers_act.append(F.relu(activation[key])* do_masks[cnt_act]) # Note: we need to register the activations taking into account non-linearity and dropout mask
                #mod_layers_act.append(F.relu(activation[key]))
                cnt_act += 1
        mod_error = mod_outputs - target_onehot

        # compute the delta_w for the batch
        delta_w_all = []
        v_w = []
        for l_idx,w in enumerate(net.parameters()):
            v_w.append(torch.zeros(w.shape))

        for l in range(len(layers_act)):

            # update for the last layer
            if l == len(layers_act)-1:

                if len(layers_act)>1:
                    delta_w = -mod_error.T @ mod_layers_act[-2]
                else:
                    delta_w = -mod_error.T @ mod_inputs

            # update for the first layer
            elif l == 0:
                delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_inputs

            # update for the hidden layers (not first, not last)
            elif l>0 and l<len(layers_act)-1:
                delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_layers_act[l-1]

            delta_w_all.append(delta_w)

        # apply the weight change
        if optim == 'SGD':
            for l_idx,w in enumerate(net.parameters()):
                with torch.no_grad():
                    w += eta * delta_w_all[l_idx]/batch_size # specify for which layer

        elif optim == 'mom':
            for l_idx,w in enumerate(net.parameters()):
                with torch.no_grad():
                    v_w_all[l_idx] = gamma * v_w_all[l_idx] + eta * delta_w_all[l_idx]/batch_size
                    w += v_w_all[l_idx]


        # keep track of the loss
        loss = criterion(outputs, target)
        # print statistics
        running_loss += loss.item()
        if i%500 == 499:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

    print('Testing...')
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data
            test_images = torch.flatten(test_images, 1) # flatten all dimensions except batch
            # calculate outputs by running images through the network
            test_outputs = net(test_images,do_masks=None)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(test_outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

    print('Test accuracy epoch {}: {} %'.format(epoch, 100 * correct / total))
    test_accs.append(100 * correct / total)


train_error = 1-test_accs[-1]/100

print('Finished Training')
