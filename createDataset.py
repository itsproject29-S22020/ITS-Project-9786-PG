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
'''
#REFERENCING:
# Author: Aladdin Persson
# Title: Machine Learning Collection: Build a Custom Dataset
# Link: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/aba36b89b438ca8f608a186f4d61d1b60c7f24e0/ML/Pytorch/Basics/custom_dataset/custom_dataset.py#L12-L29
'''

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import PIL
from PIL import Image
from torchvision.transforms import ToTensor

class DepressionLabelled_Dataset (Dataset):
    def __init__(self, csvFile, rootDir, transform):
        self.annotations = pd.read_csv(csvFile)
        self.rootDir = rootDir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations) #13530 images in total
    
    def __getitem__(self, index):
        filePath = os.path.join(self.rootDir, self.annotations.iloc[index, 0])
        img = io.imread(filePath)
        img = Image.open(filePath)
        img = img.resize((224,224), Image.ANTIALIAS)
        
        img = ToTensor()(img).unsqueeze(0)
        
        y_label = int(self.annotations.iloc[index, 1])
        
        if self.transform:
            img = self.transform(img)
            
        return(img, y_label)
    