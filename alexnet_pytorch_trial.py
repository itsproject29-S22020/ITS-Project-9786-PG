# -*- coding: utf-8 -*-
'''
### UNIT 9786 PG: ITS CAPSTONE PROJECT - UNIVERSITY OF CANBERRA
### ITS PROJECT 29-S2: Depression Analysis from Facial Video Data via Deep Learning
### PROJECT TEAM MEMBERS:
# Hang Hoang - u3197442
# Charmane Foo - u3201698
# Matt Lally - u3167761
# Lakmal Attanayake - u3177896

'''

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader


from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage import io
import pandas as pd
import numpy as np

import torch.nn.functional as F

filePath = "C:\\Users\\Hang\\OneDrive - University of Canberra\\Tech Proj\\OneDrive_2_12-10-2020\\ITS_Project_2\\Faces-combined - Copy\\combined1"
rnd_seed = 42


# Create a dataset with the corresponding label column from the csv file
class DepressionLabelled_Dataset(Dataset):
    def __init__(self, csvFile, rootDir, transform):
        self.annotations = pd.read_csv(csvFile)
        self.rootDir = rootDir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations) #13530 images in total
    
    def __getitem__(self, index):
        filePath = os.path.join(self.rootDir, self.annotations.iloc[index, 0])
        #img = io.imread(filePath)
        #img = PIL.Image.open(filePath)
        img = Image.open(filePath)
        label = int(self.annotations.iloc[index, 1])
        
        if self.transform:
            img = self.transform(img)
            
        return(img, label)

# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set parameters
input_channels = 3
no_classes = 2
learningRate = 0.001
batchSize = 16
no_epochs = 15
rnd_seed = 42

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

# Load dataset with labels
dataset = DepressionLabelled_Dataset(csvFile = 'labelled_dataset.csv', 
                                     rootDir = 'faces_extracted', 
                                     transform=torchvision.transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(), norm]))
                                     #transform = transforms.ToTensor())
                                     #transform=torchvision.transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]))

                                    
                                 
ds_size = len(dataset)

trainSize = int(0.8 * ds_size)
testSize = ds_size - trainSize

#trainSet, testSet = torch.utils.data.random_split(dataset,[10824, 2706])
trainSet, testSet = torch.utils.data.random_split(dataset,[trainSize, testSize])
trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, num_workers = 0)
testLoader = torch.utils.data.DataLoader(dataset=testSet, batch_size=batchSize, shuffle=True, num_workers = 0)

# Checking the train & test set
for faceImg, labels in trainLoader:
    print("Dimensions of train set image batch: ", faceImg.shape)
    print("Dimensions of train set image label: ", labels.shape)
    break

for faceImg, labels in testLoader:
    print("Dimensions of test set image batch: ", faceImg.shape)
    print("Dimensions of test set image label: ", labels.shape)
    break

torch.manual_seed(rnd_seed)
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = learningRate)
criterion = nn.NLLLoss()

# Define training

def check_accuracy (model, dataLoader, device):
    correct_prediction = 0
    no_examples = 0
    
    for k, (classes, targets) in enumerate(dataLoader):
        classes = classes.to(device)
        #l, p = model(classes)
        #_, predicted_classes = torch.max(p,1) #Returns the maximum value of all elements in the input tensor
        scores = model(classes)
        _, predicted_classes = scores.max(1)
        no_examples += targets.size(0)
        assert predicted_classes.size() == targets.size()
        correct_prediction += (predicted_classes == targets).sum()
    return (correct_prediction.float()/no_examples)*100


loss_list = []
train_accuracy = []
test_accuracy = []


for epoch in range(no_epochs):
    model.train()
    for batch_index, (classes, targets) in enumerate (trainLoader):
        classes = classes.to(device)
        targets = targets.to(device)
        
        # Forward and backpropagation
        #l, p = model(classes)
        #loss = F.cross_entropy(l, targets)
        scores = model(classes)
        loss = criterion(scores, targets)
        #loss = F.cross_entropy(scores, targets)
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        loss_list.append(loss.item())
        print("Epoch: {0}  | Loss: {1}".format(epoch+1,loss))
        
    model.eval()
    
    with torch.set_grad_enabled(False):
        trainAcc = check_accuracy(model, trainLoader, device)
        testAcc = check_accuracy(model, testLoader, device)
        
        print("Epoch: {0} | Total Epochs: {1}".format(epoch+1, no_epochs))
        print("Train Accuracy: {0} | Test Accuracy: {1}".format(trainAcc, testAcc))
        
        train_accuracy.append(trainAcc)
        test_accuracy.append(testAcc)
        
#model.eval()

     
        

