'''
### UNIT 9786 PG: ITS CAPSTONE PROJECT - UNIVERSITY OF CANBERRA
### ITS PROJECT 29-S2: Depression Analysis from Facial Video Data via Deep Learning
### PROJECT TEAM MEMBERS:
# Hang Hoang - u3197442
# Charmane Foo - u3201698
# Matt Lally - u3167761
# Lakmal Attanayake - u3177896
'''


#from PIL import Image
#import numpy as np 
#import sys
import os
import csv
import pandas as pd



# load the original image

path3 = 'C:\\Users\\Hang\\OneDrive - University of Canberra\\Tech Proj\\OneDrive_2_12-10-2020\\ITS_Project_2\\Faces-combined\\combined1'
#myFileList = createFileList(path3)



#from tkinter import Tcl
# https://github.com/RohitMidha23/Image-Directory-to-CSV/blob/master/img_to_CSV.py#L35
# default format can be changed as needed
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=True):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(name)
                fileList.append(fullName)
    return fileList



import natsort
f = open("trial.csv",'w', newline='') 
with f:
    writer = csv.writer(f)
    for root, dirs, fileNames in os.walk(path3, topdown=False):
        l = os.listdir(path3)
        l_sorted = natsort.natsorted(l)        
        
        for fileNames in l_sorted:
            #print (fileNames)
            writer.writerow([fileNames])
            

df = pd.read_csv('trial.csv', header=None)
df.rename(columns={0:'imageID'}, inplace=True)
df["label"]=""

df.iloc[0:300,1] = 0
df.iloc[301:390,1] = 0
df.iloc[391:459,1] = 0
df.iloc[460:527,1] = 0
df.iloc[528:828,1] = 0
df.iloc[829:1026,1] = 0
df.iloc[1027:1328,1] = 0
df.iloc[1329:1630,1] = 1
df.iloc[1631:1919,1] = 1
df.iloc[1920:2179,1] = 0
df.iloc[2180:2473,1] = 1
df.iloc[2474:2541,1] = 0
df.iloc[2542:2808,1] = 1
df.iloc[2809:3109,1] = 1
df.iloc[3110:3410,1] = 1
df.iloc[3411:3707,1] = 0
df.iloc[3708:4008,1] = 0
df.iloc[4009:4298,1] = 0
df.iloc[4299:4568,1] = 0
df.iloc[4569:4870,1] = 0
df.iloc[4871:5172,1] = 1
df.iloc[5173:5474,1] = 0
df.iloc[5475:5775,1] = 1
df.iloc[5776:6076,1] = 1
df.iloc[6077:6378,1] = 0
df.iloc[6379:6680,1] = 1
df.iloc[6681:6981,1] = 1
df.iloc[6982:7283,1] = 1
df.iloc[7284:7585,1] = 1
df.iloc[7586:7887,1] = 0
df.iloc[7888:8160,1] = 0
df.iloc[8161:8180,1] = 1
df.iloc[8181:8480,1] = 1
df.iloc[8482:8777,1] = 0
df.iloc[8778:9079,1] = 1
df.iloc[9080:9381,1] = 1
df.iloc[9382:9683,1] = 1
df.iloc[9684:9984,1] = 1
df.iloc[9985:10285,1] = 0
df.iloc[10286:10587,1] = 0
df.iloc[10588:10863,1] = 0
df.iloc[10864:11165,1] = 0
df.iloc[11166:11461,1] = 0
df.iloc[11462:11760,1] = 0
df.iloc[11761:12061,1] = 0
df.iloc[12062:12363,1] = 1
df.iloc[12364:12629,1] = 0
df.iloc[12630:12926,1] = 0
df.iloc[12927:13228,1] = 0
df.iloc[13229:13530,1] = 0

df.iat[1329,1] = 1
#print(df.at[1329,'label'])


df.to_csv('labelled_dataset.csv', index=False)
    




