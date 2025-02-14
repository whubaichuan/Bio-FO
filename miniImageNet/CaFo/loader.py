"""
Implementation of dataloader
"""
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize,RandomRotation,RandomHorizontalFlip,CenterCrop



def load_data_cafo(args):
    if args.data == 'CIFAR10':
        train_loader, test_loader = CIFAR10_loaders()
        num_classes, input_channels, input_size=10,3,32
    elif args.data == 'CIFAR100':
        train_loader, test_loader = CIFAR100_loaders()
        num_classes, input_channels, input_size = 100, 3, 32
    elif args.data == 'MNIST':
        train_loader, test_loader = MNIST_loaders()
        num_classes, input_channels, input_size = 10, 1, 28
    elif args.data == 'ImageNet':
        train_loader, test_loader = ImageNet_loaders()
        num_classes, input_channels, input_size = 100, 3, 84
    else:
        raise Exception
    return train_loader,test_loader,num_classes,input_channels,input_size







def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        #    transforms.Resize(224),
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def CIFAR10_loaders(train_batch_size=50000, test_batch_size=10000):
    transform_train = transforms.Compose([
        # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
        #    transforms.RandomCrop(32, padding=4),
        # 按0.5的概率水平翻转图片
        #    transforms.RandomHorizontalFlip(),

        #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10('./data/', train=True,
                                            download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10('./data/', train=False,
                                           download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def CIFAR100_loaders(train_batch_size=50000, test_batch_size=10000):
    transform_train = transforms.Compose([
        # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
        #    transforms.RandomCrop(32, padding=4),
        # 按0.5的概率水平翻转图片
        #    transforms.RandomHorizontalFlip(),

        #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR100('./data/', train=True,
                                             download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR100('./data/', train=False,
                                            download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader



def ImageNet_loaders(train_batch_size=51000, test_batch_size=9000):


    transform=transforms.Compose([
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

    #print(n_train)
    #print(n_test)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid, n_test])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    

    return train_loader, test_loader



