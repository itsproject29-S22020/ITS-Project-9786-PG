# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:32:59 2020

@author: Matt Lally
"""
import cv2
import glob
import os
import torchvision.transforms as transforms
from PIL import Image

#edit path
main_path = "C:\\Users\\TechFast Australia\\Desktop\\SEM 2 2020"
input_video_path = main_path + "\\depression_videos\\Video"
subclip_video_path = main_path + "\\subclips"
video_folder = main_path + "\\few_videos\\"
frames_folder = main_path + "\\frames"
cascPath = r'C:/Users/TechFast Australia/Desktop/stuff/haarcascade_frontalface_default.xml' 

#declare variable to track image numbers
i=0

#set the cascades classifers
faceCascade = cv2.CascadeClassifier(cascPath)

#counter for the subject number
count = 0 

#loop through the files in this folder
for files in os.walk(frames_folder):
    
        #counter for image number
        i=0
        count+=1
        
        
        #save in each png image in an array called images
        images = glob.glob(frames_folder + "\\frames_video{0}\\*.png".format(count))
        
        #for each image in the array images
        for image in images:
            
        #read and convert image
            try:
                #read in the image
                img = cv2.imread(image)
                
                #change the image to greyscale
                grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
                
                #detect faces using the cascade classifier loaded in with the parameters listed below
                faces = faceCascade.detectMultiScale(
                    grey,
                    scaleFactor=1.01,
                    minNeighbors=4,
                    minSize=(30,30),
                    )

                
            except:
                print("not working")
                   
            try:
                    
                print("Found {0} faces!".format(len(faces)))
        
                #check the number of faces detected
                if len(faces) == 1:
                    #load in the face into postion 0 in the faces array
                    x,y,w,h = faces[0]
                    #crop the image, based on the detected face coordinates
                    cropedImage = img[ y:y+h, x:x+w]
                    i+=1
                    
                    #create a folder if one doesn't exist and save the image that shows the cropped face in it.
                    facesfolder = frames_folder +"\\frames_video{0}\\faces{0}".format(count)
                    if not os.path.exists(facesfolder):
                           os.makedirs(facesfolder)
                    cv2.imwrite(facesfolder+ "\\{0}.jpg".format(i),cropedImage)
                    #print("face saved") 
                
                #if the numbers of dected faces is more than one, crop the image based on the last face detected. 
                elif len(faces) > 1:
                    
                    for(x,y,w,h) in faces:
                        cropedImage = img[ y:y+h, x:x+w]
                    i+=1

                    #save create a folder if one doesn't exist and save the image that shows the cropped face in it.
                    facesfolder = frames_folder +"\\frames_video{0}\\faces{0}".format(count)
                    if not os.path.exists(facesfolder):
                           os.makedirs(facesfolder)
                    cv2.imwrite(facesfolder+ "\\{0}.jpg".format(i),cropedImage)
                    
            
            except:
                print('not working 2')
             
