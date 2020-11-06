### UNIT 9786 PG: ITS CAPSTONE PROJECT - UNIVERSITY OF CANBERRA
### ITS PROJECT 29-S2: Depression Analysis from Facial Video Data via Deep Learning
### PROJECT TEAM MEMBERS:
# Hang Hoang - u3197442
# Charmane Foo - u3201698
# Matt Lally - u3167761
# Lakmal Attanayake - u3177896

import os
import moviepy
import cv2
import shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from glob import glob


#from moviepy.video.io.VideoFileClip import VideoFileClip

#edit path
main_path = "C:\\Users\\charm\\Desktop\\SEM 2 2020\\ITS_Project_2"
input_video_path = main_path + "\\depression_videos\\development\\video"
subclip_video_path = main_path + "\\subclips"
video_folder = main_path + "\\few_videos"
frames_folder = main_path + "\\frames"


#create times.txt file in main folder
os.chdir(main_path)
subclipTimes = open("times.txt", 'w')

#edit video time to subclip
subclipTimes.write("0-10")
subclipTimes.close()

with open("times.txt") as t:
    times = t.readlines()

x = 0

times = [x.strip() for x in times]

#change directory to video folder and extract subclip
os.chdir(video_folder)
for file in os.listdir(video_folder):
    for time in times:
        starttime = int(time.split("-")[0])
        endtime = int(time.split("-")[1])
        subjectSubclipFolder = "subclip-" + str(file)
        ffmpeg_extract_subclip(file, starttime, endtime, targetname=(str('subclip') + str(file)))

#move subclips to different folder
files = os.listdir(video_folder)
for f in files: 
    if f.startswith("subclip"):
        shutil.move(f, subclip_video_path)

#extract frames from all subclips and place them in individual folders
sub_clips_path = glob(subclip_video_path + "//*.mp4")

for count, sub_clips in enumerate(sub_clips_path):
    currentVid = cv2.VideoCapture(sub_clips)
    
    img_count = 0
    
    while currentVid.isOpened():
        success, image = currentVid.read()
        
        if success:
            newFrameFolder = "frames_video{}".format(count + 1)
            
            if not os.path.exists(newFrameFolder):
                os.makedirs(newFrameFolder)
                
            img_name = os.path.join(newFrameFolder, "frame{}.png".format(img_count + 1))
            cv2.imwrite(img_name, image)
            img_count += 1
        
        else:
            break

    #close video capture
    currentVid.release()  

#move folders of frames in subclip to frames folder in main path
frames = os.listdir(video_folder)
for f in frames: 
    if f.startswith("frames"):
        shutil.move(f, frames_folder)


