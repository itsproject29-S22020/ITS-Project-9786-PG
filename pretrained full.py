# -*- coding: utf-8 -*-
"""
### UNIT 9786 PG: ITS CAPSTONE PROJECT - UNIVERSITY OF CANBERRA
### ITS PROJECT 29-S2: Depression Analysis from Facial Video Data via Deep Learning
### PROJECT TEAM MEMBERS:
# Hang Hoang - u3197442
# Charmane Foo - u3201698
# Matt Lally - u3167761
# Lakmal Attanayake - u3177896

"""

from PIL import Image
# Imports
import torch
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision
import os
import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt

class DepressionLabelled_Dataset(Dataset):
    def __init__(self, csvFile, rootDir, transform):
        self.annotations = pd.read_csv(csvFile)
        self.rootDir = rootDir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations) #13530 images in total
    
    def __getitem__(self, index):
        filePath = os.path.join(self.rootDir, self.annotations.iloc[index, 0])
        img = io.imread(filePath)
        #img = PIL.Image.open(filePath)
        y_label = int(self.annotations.iloc[index, 1])
        y_label = torch.tensor(y_label, dtype = torch.float32)
        
        if self.transform:
            img = self.transform(img)
            
        return(img, y_label)
    
# Choose device set to cuda if possible
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
input_channels = 3
no_classes = 2
learningRate = 0.00001
batchSize = 32
no_epochs = 20
rnd_seed = 42



norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

# Load dataset with labels
'''
# MAJOR CODING ERRORS: THE TRAINING & TESTING SETS WERE NOT ORGANIZED BY SUBJECTS, LEADING TO ABNORMALLY HIGH ACCURACIES 
dataset = DepressionLabelled_Dataset(csvFile = 'labelled_dataset.csv', 
                                     rootDir = 'faces_extracted', 
                                     transform=torchvision.transforms.Compose([transforms.ToTensor(), norm]))

#trainSet, testSet = torch.utils.data.random_split(dataset,[10824, 2706])
ds_size = len(dataset)
trainSize = int(0.8 * ds_size)
testSize = ds_size - trainSize

trainSet, testSet = torch.utils.data.random_split(dataset,[trainSize, testSize])
trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testSet, batch_size=batchSize, shuffle=True)
'''

# Load dataset with labels
# For our quick fixes, we decided to split the train and test sets into two different folders, this will at least make sure the model is tested on unseen data
# Future work could focus on 10-fold cross validation to split the train and test sets
trainSet = DepressionLabelled_Dataset(csvFile = 'C:\\Users\\TechFast Australia\\Desktop\\blank\\train_labels.csv', 
                                     rootDir = 'C:\\Users\\TechFast Australia\\Desktop\\blank\\train_set', 
                                     transform=torchvision.transforms.Compose([transforms.ToTensor(), norm]))
                                     #transform = transforms.ToTensor())
                                     #transform=torchvision.transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]))
testSet = DepressionLabelled_Dataset(csvFile = 'C:\\Users\\TechFast Australia\\Desktop\\blank\\test_labels.csv', 
                                     rootDir = 'C:\\Users\\TechFast Australia\\Desktop\\blank\\test_set', 
                                     transform=torchvision.transforms.Compose([transforms.ToTensor(), norm]))

trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testSet, batch_size=batchSize, shuffle=True)


# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

# Load pretrained ResNet50 Model
torch.manual_seed(rnd_seed) # set the random seed

model = torchvision.models.resnet50(pretrained=True)
model.to(device)

# modify the fully connected layer of the network
model.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid())

# Loss and Optimizer Function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learningRate)

train_list = {"Loss" : [], "Accuracy" : []}
test_list = {"Loss" : [], "Accuracy" : []}


# Determine accuracy
def compute_accuracy (predictions, labels):
    for i in range (len(predictions)):
        if predictions[i] >= 0.5: 
            predictions = 1
        else: 
            predictions = 0
    
    for i in range (len(predictions)):
        if predictions[i] == labels[i]:
            accuracy = 1
        else:
            accuracy = 0
    
    accuracy = np.sum(accuracy)/len(predictions)
    accuracyP = accuracy * 100
    return accuracyP

# Define training of the network
def compute_train(trainLoader):
    
    loss_list = []
    acc_list = []
    
    # Iterate over train dataloader
    for facedata, labels in trainLoader:
        
        facedata = facedata.to(device)
        labels = labels.to(device)
        
        # Reset the gradients
        optimizer.zero_grad()
        
        # Forward inputs
        predictions = model(facedata)
        
        # Compute loss function
        _loss = criterion(predictions, labels)
        loss = _loss.item()
        loss_list.append(loss)
        
        # Backpropagation
        _loss.backward()
        optimizer.step()
          
# Overall accuracy and loss of each epoch
    eLoss = np.mean(loss_list)
    eAccuracy = np.mean(acc_list)
    
# Store train results for logging
    train_list["loss"].append(eLoss)
    train_list["accuracy"].append(eAccuracy)
    
    return eLoss, eAccuracy

# Define testing of the network
def compute_test(testLoader, best_test_acc):
    loss_list = []
    acc_list = []
    
    for facedata, labels in testLoader:
        
        facedata = facedata.to(device)
        labels = labels.to(device)
        
        # Reset the gradients
        optimizer.zero_grad()
        
        # Forward inputs
        predictions = model(facedata)
        
        # Compute loss function
        _loss = criterion(predictions, labels)
        loss = _loss.item()
        loss_list.append(loss)
        
# Overall accuracy and loss of each epoch
    eLoss = np.mean(loss_list)
    eAccuracy = np.mean(acc_list)
    
# Store train results for logging
    test_list["loss"].append(eLoss)
    test_list["accuracy"].append(eAccuracy)
    
# Save the best model:
    if eAccuracy > best_test_acc:
        best_test_acc = eAccuracy
        torch.save(model.state_dict(), "best_resnet50.pth")
        
    return eLoss, eAccuracy, best_test_acc


# Train and Test the network
best_test_acc = 0
for epoch in range(no_epochs):
    
    loss, accuracy = compute_train(trainLoader)
    
    print ("Training Network:")
    print("\nEpoch {0}".format(epoch+1))
    print("Training Loss: {0}".format(round(loss,3)))
    print("Training Accuracy: {0}".format(round(accuracy,3)))
    
    loss, accuracy, best_test_acc = compute_test(testLoader, best_test_acc)
    
    print ("Testing Network:")
    print("\nEpoch {0}".format(epoch+1))
    print("Testing Loss: {0}".format(round(loss,3)))
    print("Testing Accuracy: {0}".format(round(accuracy,3)))

# Plots for accuracy and loss

# Accuracy
plt.title("Overall Training & Testing Accuracy")    
plt.plot(np.arrange(1, 11, 1), train_list["accuracy"], color = 'red')
plt.plot(np.arrage(1, 11, 1), test_list["accuracy"], color = 'blue')
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.show()

# Loss
plt.title("Overall Training & Testing Loss")    
plt.plot(np.arrange(1, 21, 1), train_list["loss"], color = 'red')
plt.plot(np.arrage(1, 21, 1), test_list["loss"], color = 'blue')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.show()
