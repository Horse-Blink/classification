# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report
from collections import deque
import pandas as pd

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    
    Compose,
    LoadImage,
    Resize,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)
from monai.utils import set_determinism
import cv2
import argparse
import sys

parser=argparse.ArgumentParser()

parser.add_argument("--path", default="./videos/GH010022.MP4",type=str,help="path for video")
parser.add_argument("--output",default="./FrameSave/",type=str,help="where to save the output")

args=parser.parse_args()

print(' '.join(sys.argv))


set_determinism(seed=0)

class_names = ["Eyes Open","Blinking"]

df=pd.read_excel("./Blink rate automation.xlsx",sheet_name=None) # reading in from excel as dict of dataframes
#trivia: locals()["str"] = whatever to turn str into variable called str
df=dict(sorted(df.items())) # sorting the annotations in alphabetical order
df['Guide']['Video']=df['Guide']['Video'].map(lambda x : x.rstrip())
df["GH010022.MP4"]=df["GH010022.MP4"][df["GH010022.MP4"]['Class(F=Full,H=Half)']=='F'].reset_index()

#get video length from the excel file
video_length=int(df['Guide']['Total frame count'].loc[df['Guide']['Video']=="GH010022.MP4"])

labels_dict=dict()
#set up an array to save blinking labels- this is later used to extract blink frames 
ground_truth=np.zeros(video_length+1)

for row in range(len(df["GH010022.MP4"])):
    # print(row)
    ground_truth[int(df["GH010022.MP4"]['Eye Closed Frame'][row]) : int(df["GH010022.MP4"]['Eye opening'][row])]=1
    




num_class = len(class_names) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DenseNet121(spatial_dims=2, in_channels=3,
                    out_channels=num_class).to(device)
model_name="best_metric_model2022-06-06DenseNet121f1s2"#"2022-05-12T22zoo_avg"
model=torch.nn.DataParallel(model)
model.load_state_dict(torch.load( "./"+model_name))


vc_obj=cv2.VideoCapture(args.path)
W,H=None,None

video_length = int(vc_obj.get(cv2.CAP_PROP_FRAME_COUNT)) -1 # -1 should give exact length but sometimes some frames are missing which causes endless looping
print("video length", video_length)
Q=deque(maxlen=10)
pred_label=[]
pred_label_processed=[]
ind=[]
conv_ar0=[]
conv_ar1=[]
while vc_obj.isOpened():
    writer=None
    for i in range(video_length):
        rtn_flag,frame=vc_obj.read()
        if not rtn_flag:
            continue
        
        if W is None or H is None:
            (H,W)=frame.shape[:2]
        print(frame.shape)    
        image=cv2.resize(frame,(480,360),interpolation = cv2.INTER_AREA)
        output=image.copy()
        #frame=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image1=torch.zeros((3,360,480))
        # image1=torch.tensor(cv2.resize(image2,(480,360), interpolation=cv2.INTER_AREA))
        image1[0,:,:]=torch.tensor(cv2.resize(np.array(image[0,:,:]),(480,360),interpolation=cv2.INTER_AREA))
        image1[1,:,:]=torch.tensor(cv2.resize(np.array(image[1,:,:]),(480,360),interpolation=cv2.INTER_AREA))
        image1[2,:,:]=torch.tensor(cv2.resize(np.array(image[2,:,:]),(480,360),interpolation=cv2.INTER_AREA))
        
        
        with torch.no_grad():
            frame=image1.to(device)#torch.tensor(image).to(device)
            #frame=frame.moveaxis(-1,0)
            frame=frame[None,:].float()
          
            print(frame.shape)
            pred=model(frame)[0].argmax(dim=1)
            print(pred)
            conv_ar0.append(pred[0].cpu().numpy())
            
            conv_ar1.append(pred[1].cpu().numpy())
            np_pred=np.argmax(pred.cpu().numpy())         
            # print(np_pred)
            pred_label.append(np_pred)
            Q.append(pred) # queue object only stores maxlen objects then ejects the oldest 
            # print(np.array(Q)[0].shape)
            npQ=np.array([j.cpu().numpy() for j in Q])
            # print(type(npQ),npQ.shape,npQ)
            result=npQ.mean(axis=0)
            
            # print(result.shape)
             
            blink=np.argmax(result)
            
         
                        
            label=class_names[blink]
            
           
            pred_label_processed.append(blink)
            ind.append(i)
            
        
        
        
        # #draw the activity on output frame
        # text="{}".format(label)
        # cv2.putText(output,text,(35,50),cv2.FONT_HERSHEY_SIMPLEX,1.25,(0,255,0),5)
        
        # cv2.imwrite(args.output+str(i)+".jpg",output)
        
        # if writer is None:
        #     four_cc=cv2.VideoWriter_fourcc(*"MJPG")
        #     writer=cv2.VideoWriter(args.output,four_cc,30,(W,H),True)
            
        # writer.write(output)
        print(i,"/",video_length)
    
    break
        # cv2.imshow("HorseCare",output)
        # key=cv2.waitKey(1) & 0xFF
        
        # #if q pressed break from loop
        # if key==ord("q"):
        #     break
    
            
pred_array=np.array(pred_label)
kernel=np.ones(10)
conv_avg0=np.convolve(np.array(conv_ar0), kernel,'same')/len(kernel)
conv_avg1=np.convolve(np.array(conv_ar1), kernel,'same')/len(kernel)
conv_avg=np.stack((conv_avg0,conv_avg1))
conv_pred=np.argmax(conv_avg, axis=0)

# release the file pointers
df=pd.DataFrame()
df["Human label"]=ground_truth[ind]
df["Frame-wise Pred"]=pred_label
df["Moving average"]=pred_label_processed
df["Conv processed"]=conv_pred

df.to_csv(f"./GH010022_{model_name}pred{len(Q)}.csv")

print("[INFO] cleaning up...")
# writer.release()
# vc_obj.release()           
        
        
    


