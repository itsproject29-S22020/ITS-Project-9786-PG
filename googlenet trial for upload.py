
### UNIT 9786 PG: ITS CAPSTONE PROJECT - UNIVERSITY OF CANBERRA
### ITS PROJECT 29-S2: Depression Analysis from Facial Video Data via Deep Learning
#code from:
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/aba36b89b438ca8f608a186f4d61d1b60c7f24e0/ML/Pytorch/Basics/custom_dataset/custom_dataset.py#L12-L29
#amended by:
### PROJECT TEAM MEMBERS:
# Hang Hoang - u3197442
# Charmane Foo - u3201698
# Matt Lally - u3167761
# Lakmal Attanayake - u3177896
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.utils.data import DataLoader

#from createDataset import DepressionLabelled_Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pandas as pd
from skimage import io

#define dataset class with images and labels
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
        
        if self.transform:
            img = self.transform(img)
            
        return(img, y_label)
    
# Choose device
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set parameters
input_channels = 3
no_classes = 2
learningRate = 0.000001
batchSize = 16
no_epochs = 5
rnd_seed = 35

#normalise the images inline with GoogLeNet requirements
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

# Load dataset with labels, split between trainset and testset
trainSet = DepressionLabelled_Dataset(csvFile = 'C:\\Users\\TechFast Australia\\Desktop\\blank\\train_labels.csv', 
                                     rootDir = 'C:\\Users\\TechFast Australia\\Desktop\\blank\\train_set', 
                                     transform=torchvision.transforms.Compose([transforms.ToTensor(), norm]))
                                     #transform = transforms.ToTensor())
                                     #transform=torchvision.transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]))
testSet = DepressionLabelled_Dataset(csvFile = 'C:\\Users\\TechFast Australia\\Desktop\\blank\\test_labels.csv', 
                                     rootDir = 'C:\\Users\\TechFast Australia\\Desktop\\blank\\test_set', 
                                     transform=torchvision.transforms.Compose([transforms.ToTensor(), norm]))
                                    
                                 
#ds_size = len(dataset)

#load the data into the data loader
trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testSet, batch_size=batchSize, shuffle=True)

torch.manual_seed(rnd_seed) # set the random seed

# Load the pre-trained model
model = torchvision.models.googlenet(pretrained=True)
model.to(dev)

# Loss and Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learningRate)


# Train the network
for epoch in range(no_epochs):
    loss_list = []
    
    for batch_index, (facedata, targetlabel) in enumerate(trainLoader):
        # Use CUDA if available
        facedata = facedata.to(device=dev)
        targetlabel = targetlabel.to(device=dev)
        
        # Forward and backpropagation
        score = model(facedata)
        loss = criterion(score, targetlabel)
        
        loss_list.append(loss.item())
        
        # Reset the gradients
        optimizer.zero_grad()
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        cost = sum(loss_list)/len(loss_list)
    #print(f'Cost at epoch {epoch} is {sum(loss_list)/len(loss_list)}')
    print('The loss at epoch no {0} is {1}'.format(epoch+1,cost))
         
# Check training and testing accuracy overall to see how our model performs
def model_accuracy_check(data_loader, model):
    no_correct_pred = 0
    no_trials = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device=dev)
            y = y.to(device=dev)
            
            score = model(x)
            _, predictions = score.max(1)
            no_correct_pred += (predictions == y).sum()
            no_trials += predictions.size(0)
        
        print(f'The model had {no_correct_pred} / {no_trials} images with an accuracy of {float(no_correct_pred)/float(no_trials)*100:.2f}') 
    
    model.train()

print("The Training Set Accuracy:")
model_accuracy_check(trainLoader, model)

print("The Testing Set Accuracy:")
model_accuracy_check(testLoader, model)
