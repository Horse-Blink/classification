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

parser.add_argument("--batch_size", default=1, type=int,help="number of images to process for each step of gradient descent")
parser.add_argument("--model",default="DenseNet121",type=str, help="DenseNet121 or DenseNet264")
parser.add_argument("--load_save",default=0, type=int,help="load saved weights from previous training")
parser.add_argument("--load_name",default='best_metric_modelDenseNet121auc.pth',type=str,help="name of model to be loaded")
parser.add_argument("--epochs", default=100, type=int, help="number of epochs to run")
parser.add_argument("--opt",default="acc",type=str, help= "Optimisation metric to use- 'auc' or 'acc'")
parser.add_argument("--comb_ch",default=1,type=int, help="train with all 3 channels or combine them into 1")
parser.add_argument("--eval_only",default=0,type=int,help="Eval only with loaded model")
parser.add_argument("--eval_model",default="best_metric_model2022-03-16DenseNet121acc.pth",type=str, help="model to be loaded for eval")
parser.add_argument("--method", default="std", type=str, help="save str to diff models while training in parallel")
parser.add_argument("--seed", default="0", type=int, help="random seed")

args=parser.parse_args()

print(' '.join(sys.argv))


set_determinism(seed=0)

class_names = ["Eyes Open","Blinking"]

num_class = len(class_names) 

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

image_files=[os.path.join('./All Frames','GH010022',x) for x in os.listdir('./All Frames/GH010022/')]
label_path='./All Frames/GH010022/label.npy'
image_class=np.load(label_path)

test_x=image_files
test_y=image_class

val_transforms = Compose(
    [LoadImage(image_only=True),  ScaleIntensity(), EnsureType()])
    
y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image= self.transforms(self.image_files[index])
        if image.shape[1]==1440:
            image=torch.movedim(image,1,0)            
            
        
       
        image= torch.movedim(image,-1,0)  
        
       
        image1=torch.zeros((3,360,480))
        # image1=torch.tensor(cv2.resize(image2,(480,360), interpolation=cv2.INTER_AREA))
        image1[0,:,:]=torch.tensor(cv2.resize(np.array(image[0,:,:]),(480,360),interpolation=cv2.INTER_AREA))
        image1[1,:,:]=torch.tensor(cv2.resize(np.array(image[1,:,:]),(480,360),interpolation=cv2.INTER_AREA))
        image1[2,:,:]=torch.tensor(cv2.resize(np.array(image[2,:,:]),(480,360),interpolation=cv2.INTER_AREA))
        print('image1.shape=',image1.shape)
        
        if args.comb_ch==1:     
            image=image[0,:,:]*0.114+image[1,:,:]*0.587+image[2,:,:]*0.299
            image=cv2.resize(np.array(image),(360,480),interpolation=cv2.INTER_AREA)
                
            image=image[None,:,:]
            print(image.shape)
       
            
        return image1, self.labels[index]
        
test_ds = MedNISTDataset(test_x, test_y, val_transforms)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=1, num_workers=4)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.model=="EfficientNetBN":
        model = locals()[args.model]("efficientnet-b0",spatial_dims=2, in_channels=in_ch,
                            num_classes=num_class,pretrained=False).to(device)
else:
    model = locals()[args.model](spatial_dims=2, in_channels=in_ch,out_channels=num_class,pretrained=False).to(device)
model=torch.nn.DataParallel(model)  
model.load_state_dict(torch.load('./'+args.eval_model))
     
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        
        pred = model(test_images).argmax(dim=1)
        print(pred, len(test_loader)
        
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())
        
y_pred_avg=np.convolve(np.array(y_pred),np.ones(5),'same')
y_pred=(y_pred>0.5).astype(int)  
y_true=np.array(y_true)

print("len( y_pred)=", len( y_pred), "len(y_true)=",len(y_true))
print("type(y_pred), type(y_true)=",type(y_pred), type(y_true))
print('y_pred, y_true',y_pred, y_true)
print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4))


# release the file pointers
df=pd.DataFrame()
df["Human label"]=y_true
df["Frame-wise Pred"]=y_pred
df["Conv Moving average"]=y_pred_avg


df.to_csv(f"./GH010022_{args.eval_model}pred{len(Q)}.csv")

print("[INFO] cleaning up...")
# writer.release()
# vc_obj.release()           
        
        
    


