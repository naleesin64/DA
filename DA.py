### Import Necessary Modules
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")
import os

### Initialize Random Seed and different Augmentations
def random_affine_with_seed(degrees, translate, scale, shear, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    affine = transforms.RandomAffine(degrees, translate, scale, shear)
    return affine

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

degrees=([-20,20])
translate=([0.35,0.65])
scale=([0.9,1.1])
shear=([0,0])
seed = 0
torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

random_affine = random_affine_with_seed(degrees, translate, scale, shear, seed)
random_affine2 = random_affine_with_seed([-5,5],[0.45,0.55],[0.98,1.02],[0,0],0)

prototype_transform1 = transforms.Compose([random_affine, transforms.ToTensor()])
prototype_transform2 = transforms.Compose([random_affine2, transforms.ToTensor()])

random_affine3 = random_affine_with_seed([-15,15],[0,0],[0.5,1.5],[-50,50],10)
custom_transform = transforms.Compose([random_affine3, transforms.ToTensor()])

transform1 = transforms.Compose([transforms.RandomAffine(degrees=([-20,20]),
                                                        translate=([0,0]),
                                                        scale=([1,1]),
                                                        shear=([0,0])),
                                 transforms.ToTensor()])


transform2 = transforms.Compose([transforms.RandomAffine(degrees=([0,0]),
                                                        translate=([0.35,0.65]),
                                                        scale=([1,1]),
                                                        shear=([0,0])),
                                 transforms.ToTensor()])

transform3 = transforms.Compose([transforms.RandomAffine(degrees=([0,0]),
                                                        translate=([0,0]),
                                                        scale=([0.9,1.1]),
                                                        shear=([0,0])),
                                 transforms.ToTensor()])

transform4 = transforms.Compose([transforms.RandomAffine(degrees=([0,0]),
                                                        translate=([0,0]),
                                                        scale=([1,1]),
                                                        shear=([-20,20])),
                                 transforms.ToTensor()])

no_transform = transforms.Compose([transforms.RandomAffine(degrees=([0,0]),
                                                        translate=([0,0]),
                                                        scale=([1,1]),
                                                        shear=([0,0])),
                                 transforms.ToTensor()])


train_indices100_DA1 = []
train_indices1000_DA1 = []
train_indices10000_DA1 = []

for i in range(200):
    train_indices100_DA1.append(i)
for i in range(2000):
    train_indices1000_DA1.append(i)
for i in range(20000):
    train_indices10000_DA1.append(i)

train_indices100_DA2 = []
train_indices1000_DA2 = []
train_indices10000_DA2 = []

for i in range(300):
    train_indices100_DA2.append(i)
for i in range(3000):
    train_indices1000_DA2.append(i)
for i in range(30000):
    train_indices10000_DA2.append(i)
    
train_indices100_DA5 = []
train_indices1000_DA5 = []
train_indices10000_DA5 = []

for i in range(600):
    train_indices100_DA5.append(i)
for i in range(6000):
    train_indices1000_DA5.append(i)
for i in range(60000):
    train_indices10000_DA5.append(i)

train_indices100_DA10 = []
train_indices1000_DA10 = []
train_indices10000_DA10_1 = []
train_indices10000_DA10_2 = []

for i in range(1100):
    train_indices100_DA10.append(i)
for i in range(11000):
    train_indices1000_DA10.append(i)
for i in range(60000):
    train_indices10000_DA10_1.append(i)
for i in range(50000):
    train_indices10000_DA10_2.append(i)
train_indices10000_DA10 = np.concatenate([train_indices10000_DA10_1,train_indices10000_DA10_2])

# MNIST
# no_transform_set = torchvision.datasets.MNIST("./data", download=True, transform=no_transform)
# train_set = torchvision.datasets.MNIST("./data", download=True, transform=paper_transform1)
# test_set = torchvision.datasets.MNIST("./data", download=True, train=False, transform=
#                                                transforms.Compose([transforms.ToTensor()])) 

# EMNIST
#no_transform_set = torchvision.datasets.MNIST("./data", download=True, transform=no_transform, split="balanced")
#train_set = torchvision.datasets.MNIST("./data", download=True, transform=paper_transform1, split="balanced")
#test_set = torchvision.datasets.MNIST("./data", download=True, train=False, transform=
#                                                transforms.Compose([transforms.ToTensor()]), split="balanced") 

# FashionMNIST
# no_transform_set = torchvision.datasets.MNIST("./data", download=True, transform=no_transform)
# train_set = torchvision.datasets.MNIST("./data", download=True, transform=paper_transform1)
# test_set = torchvision.datasets.MNIST("./data", download=True, train=False, transform=
#                                                transforms.Compose([transforms.ToTensor()])) 

train_set100 = Subset(train_set, train_indices100_DA1)
train_set1000 = Subset(train_set, train_indices1000_DA1)
train_set10000 = Subset(train_set, train_indices10000_DA1)

### Data Loader
batch_size = 64
train_loader100 = torch.utils.data.DataLoader(train_set100, batch_size=batch_size, shuffle=True)
train_loader1000 = torch.utils.data.DataLoader(train_set1000, batch_size=batch_size, shuffle=True)
train_loader10000 = torch.utils.data.DataLoader(train_set10000, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

### Initialize Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Initialize CNN with Tangent Propagation
class Tangent_CNN(nn.Module):
    
    def __init__(self):
        super(Tangent_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=28*28, out_features=128)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def compute_jacobian(model, image):
        output = model(image)
        jacobian = torch.zeros(output.size(-1), image.view(-1).size(-1))
        for i in range(output.size(-1)):
            grad_output = torch.zeros(*output.shape)
            grad_output[0][i] = 1
            output.backward(grad_output, retain_graph=True)
            jacobian[i] = image.grad.view(-1)
            image.grad.zero_()
        jacobian_loss = torch.norm(jacobian)
        return jacobian, jacobian_loss
      
def compute_jacobian(model, input_data):
    output = model(input_data)
    jacobian = torch.zeros(output.size(-1), input_data.view(-1).size(-1))
    for i in range(output.size(-1)):
        grad_output = torch.zeros(*output.shape)
        grad_output[0][i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = input_data.grad.view(-1)
        input_data.grad.zero_()
    jacobian_loss = torch.norm(jacobian)
    return jacobian, jacobian_loss

### Training for N=100
TAN_100 = []
def TAN_train100():
    model = Tangent_CNN()
    model.to(device)
    error = nn.CrossEntropyLoss()
    learning_rate = 0.001
    tangent_weight = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    jacobian_loss_list = []
    
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader100:
            inputs.requires_grad = True
            jacob_matrix, jacob_loss = compute_jacobian(model, inputs)
            
            output = model(inputs)
            
            loss = error(output, labels)
            loss_list.append(loss)
            total_loss = loss + tangent_weight * jacob_loss
            #total_loss = total_loss.to(torch.float32)
            #total_loss = float(total_loss)
            jacobian_loss_list.append(total_loss.item())
            optimizer.zero_grad()
            #loss.backward()
            total_loss.backward()
            optimizer.step()

    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    TAN_100.append(accuracy)
    del correct, total, accuracy
    del model
    del loss, jacob_loss, total_loss
    return TAN_100, jacobian_loss_list

### Training for N=1000
TAN_1000 = []
def TAN_train1000():
    model = Tangent_CNN()
    model.to(device)
    error = nn.CrossEntropyLoss()
    learning_rate = 0.001
    tangent_weight = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    jacobian_loss_list = []
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader1000:
            inputs.requires_grad = True
            jacob_matrix, jacob_loss = compute_jacobian(model, inputs)
            
            output = model(inputs)

            loss = error(output, labels)
            loss_list.append(loss)
            total_loss = loss + tangent_weight * jacob_loss
            jacobian_loss_list.append(total_loss)
            optimizer.zero_grad()
            #loss.backward()
            total_loss.backward()
            optimizer.step()

    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    TAN_1000.append(accuracy)
    del correct, total, accuracy
    del model
    del loss, jacob_loss, total_loss
    return TAN_1000

### Training for N=10000
TAN_10000 = []
def TAN_train10000():
    model = Tangent_CNN()
    model.to(device)
    error = nn.CrossEntropyLoss()
    learning_rate = 0.001
    tangent_weight = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    jacobian_loss_list = []
    
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader10000:
            inputs.requires_grad = True
            jacob_matrix, jacob_loss = compute_jacobian(model, inputs)
            
            output = model(inputs)

            loss = error(output, labels)
            loss_list.append(loss)
            total_loss = loss + tangent_weight * jacob_loss
            jacobian_loss_list.append(total_loss)
            optimizer.zero_grad()
            #loss.backward()
            total_loss.backward()
            optimizer.step()

    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    TAN_10000.append(accuracy)
    del correct, total, accuracy
    del model
    del loss, jacob_loss, total_loss
    return TAN_10000

### Accuracy Training
tic = time.time()
for i in range(10):
    TAN_100_ = TAN_train100()
    TAN_1000_ = TAN_train1000()
    TAN_10000_ = TAN_train10000()
toc = time.time()

### Accuracy Output
print(np.mean(TAN_100), u"\u00B1", np.std(TAN_100))
print(np.mean(TAN_1000), u"\u00B1", np.std(TAN_1000))
print(np.mean(TAN_10000), u"\u00B1", np.std(TAN_10000))
print("Time taken: ", (toc-tic)/60, "minutes")
