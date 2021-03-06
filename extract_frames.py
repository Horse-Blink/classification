# -*- coding: utf-8 -*-

import cv2
import os

import cv2
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel("./Blink rate automation.xlsx",sheet_name=None) # reading in from excel as dict of dataframes
#trivia: locals()["str"] = whatever to turn str into variable called str
df=dict(sorted(df.items())) # sorting the annotations in alphabetical order
df['Guide']['Video']=df['Guide']['Video'].map(lambda x : x.rstrip())

labels_dict=dict()
for vids in df:
    print (vids)
    
    
    if vids=="Guide":
        
        pass
    else:
        path="./videos/"+vids
        video_length=int(df['Guide']['Total frame count'].loc[df['Guide']['Video']==vids]) 
        
        cap = cv2.VideoCapture(path) 
        # Find the number of frames
        # video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 - NOT reliable last resort only!!
        
        
        
        #set up an array to save blinking labels
        labels_dict[vids]=np.zeros(video_length+1)
        
        df[vids]=df[vids][df[vids]['Class(F=Full,H=Half)']=='F'].reset_index()
        
        # loop through each df and set the array to 
        for row in range(len(df[vids])):            
            # set the eye closed frames to 1 and leave rest at 0
            labels_dict[vids][int(df[vids]['Eyes Closed Frame'][row])-2 : int(df[vids]['Eye opening'][row])+5]=1
        count=0 
        try:
            os.mkdir("./Eyes Open/")
            os.mkdir("./Blinking/")
            
            
        except:
            pass
        
        time_start = time.time()
        save_eq=0
        skip_counter=0
        while cap.isOpened():
            print(count, "/",video_length)
            flag,frame=cap.read()            
            
            
            
            if not flag:
                continue
            
            elif labels_dict[vids][count]==1:                
                cv2.imwrite("./Blink Frames/" +vids+ "%#05d.jpg" % (count+1), frame)
                save_eq+=1
            elif skip_counter<5:
                skip_counter+=1
            elif save_eq>=1:
                cv2.imwrite("./Eyes Open Frames/"+vids+ "%#05d.jpg" % (count+1), frame)
                save_eq -=1
                
            
            
                       
            count=count+1            
            if (count > (video_length-1)):
                # Log the time again
                time_end = time.time()
                # Release the feed
                cap.release()
                # Print stats
                print ("Done extracting frames.\n%d frames extracted" % count)
                print ("It took %d seconds forconversion." % (time_end-time_start))
                break  
   
        continue            
               
            

            
            




# def video_to_frames(input_loc, output_loc):
#     """Function to extract frames from input video file
#     and save them as separate frames in an output directory.
#     Args:
#         input_loc: Input video file.
#         output_loc: Output directory to save the frames.
#     Returns:
#         None
#     """
#     try:
#         os.mkdir(output_loc)
#     except OSError:
#         pass
#     # Log the time
#     time_start = time.time()
#     # Start capturing the feed by initialising VideoCapture object
#     cap = cv2.VideoCapture(input_loc) 
#     # Find the number of frames
#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
#     print ("Number of frames: ", video_length)
#     count = 0
#     print ("Converting video..\n")
#     # Start converting the video
#     while cap.isOpened():
#         # Extract the frame
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         # Write the results back to output location.
#         cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
#         count = count + 1
#         # If there are no more frames left
#         if (count > (video_length-1)):
#             # Log the time again
#             time_end = time.time()
#             # Release the feed
#             cap.release()
#             # Print stats
#             print ("Done extracting frames.\n%d frames extracted" % count)
#             print ("It took %d seconds forconversion." % (time_end-time_start))
#             break

# if __name__=="__main__":

#     input_loc = './videos/GH010022.mp4'
#     output_loc = './frames/'
#     video_to_frames(input_loc, output_loc)

# # v_names=os.listdir(r"C:\Users\b_pau\OneDrive - Aberystwyth University\05_Brats_2021\Horse Blink Tracking\OneDrive_1_2-2-2022\videos")
# # #source https://theailearner.com/2018/10/15/extracting-and-saving-video-frames-using-opencv-python/


# # for vid in v_names:
# i=0
# cap=cv2.VideoCapture("./videos/GH010022.mp4" )

# if cap.isOpened()==False:
#     print(f"Error opening vid file")
# success=True
# while success:
#     success, frame = cap.read()
#     print(i, " = ", success)
#     if success == True:        
#     #     break
#         cv2.imwrite("./frames/"+ "GH010022" +"_" +str(i)+'.jpg',frame)
#         i+=1
    
 
 


# cap.release()
# cv2.destroyAllWindows()

