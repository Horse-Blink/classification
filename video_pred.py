# -*- coding: utf-8 -*-


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
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import classification_report
from datetime import date, datetime
                   

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, DenseNet264, EfficientNetBN
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
import re
          
import argparse
import sys
from monai.utils import set_determinism
from torchsummary import summary
from torchmetrics import F1Score

# print_config()
torch.multiprocessing.set_sharing_strategy("file_system")

parser=argparse.ArgumentParser()

                                                                                             
                                                                                               

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

data_dir = './'
print(data_dir)

set_determinism(seed=args.seed)

class_names =['Eyes open orig','Eyes closed orig'] # ["Eyes Open","Blinking"]

num_class = len(class_names) 
image_files = [
    [
        os.path.join(data_dir, class_names[i], x)
        for x in os.listdir(os.path.join(data_dir, class_names[i]))
    ]
    for i in range(num_class)
] # combining all image files into a list- list of two lists of images
num_each = [len(image_files[i]) for i in range(num_class)] # number of images in each list
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i]) #joining lists together
    image_class.extend([i] * num_each[i])
num_total = len(image_class)
image_width, image_height = PIL.Image.open(image_files_list[0]).size

                                                                                                              
                                                                        
                                                                           
                                                                    
                                                                                                    

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")

##########---------------------------PLOTS------------------------#############
plt.subplots(3, 3, figsize=(8, 8))
for i, k in enumerate(np.random.randint(num_total, size=9)):
    im = PIL.Image.open(image_files_list[k])
    arr = np.array(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
plt.tight_layout()
plt.show()

##########-------------------------training split-----------------------#######
val_frac = 0.1 
test_frac = 0.1
length = len(image_files_list)

indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = [i for i,item in enumerate(image_files_list) if re.search("GH010022",item)]#indices[:176]#
val_indices = [i for i,item in enumerate(image_files_list) if re.search("GH010007",item)]#indices[test_split:val_split]#
train_indices = [i for i in indices if i not in val_indices and i not in test_indices]

                                           
                       


train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]

image_files2=[os.path.join('./All Frames','GH010022',x) for x in os.listdir('./All Frames/GH010022/')]
label_path2='./All Frames/GH010022/label.npy'
image_class2=np.load(label_path2)

print(image_class2.shape, 'image_class.shape')

print(len(image_class2), 'len(image_class)')
print(len(image_files2))

length2 = len(image_files2)
indices2 = np.arange(length2)
                         
test_x=[image_files2[i] for i in indices2]
print(test_x)
test_y=[image_class2[i] for i in indices2]


print(
    f"Training count: {len(train_x)}, Validation count: "
    f"{len(val_x)}, Test count: {len(test_x)}")

##########---------------------Transforms and Data----------------------#######
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        # AddChannel(),
        
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
    ]
)

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
        # print('image1.shape=',image1.shape)
        
        if args.comb_ch==1:     
            image=image[0,:,:]*0.114+image[1,:,:]*0.587+image[2,:,:]*0.299
            image=cv2.resize(np.array(image),(360,480),interpolation=cv2.INTER_AREA)
                
            image=image[None,:,:]
            print(image.shape)
       
            
        return image1, self.labels[index]


train_ds = MedNISTDataset(train_x, train_y, train_transforms)

in_ch=train_ds[1][0].shape[0]
print(train_ds[1][0].shape)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

val_ds = MedNISTDataset(val_x, val_y, val_transforms)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=1, num_workers=4)

test_ds = MedNISTDataset(test_x, test_y, val_transforms)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=1, num_workers=4)
    
                                                                       

##########---------------------Network & Optimiser----------------------#######
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model=="EfficientNetBN":
    model = locals()[args.model]("efficientnet-b0",spatial_dims=2, in_channels=in_ch,
                        num_classes=num_class,pretrained=False).to(device)
else:
    model = locals()[args.model](spatial_dims=2, in_channels=in_ch,
                        out_channels=num_class,pretrained=False).to(device)

with torch.cuda.amp.autocast():
    summary(model,(in_ch,360,480))

loss_function = torch.nn.CrossEntropyLoss() #NLLLoss()#
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
max_epochs = args.epochs
val_interval = 1
auc_metric = ROCAUCMetric()
f1_metric=F1Score(num_classes=2).to(device)
if args.load_save==1:
    model.load_state_dict(torch.load(
        os.path.join(data_dir,'saved models', "2022-05-12T22zoo_avg")))
        
model=torch.nn.DataParallel(model)      
        ##model.load_state_dict(torch.load( os.path.join(data_dir,'saved models', "2022-05-12T22zoo_avg")))
##########--------------------------Training----------------------------#######
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

# for batch_data in train_loader:
#         print("one step in batch")
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    # print("model trained")
    epoch_loss = 0
    step = 0
    step_start = time.time()
    for batch_data in train_loader:
        # print("one step in batch")
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        # print("ba")
        optimizer.step()
        epoch_loss += loss.item()
        if step%10==0:
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [y_trans(i) for i in decollate_batch(y)]
            # print("y onehot",y_onehot[0].shape)
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)] #softmax #should be a list of tensors
            
            y_pred_act_np=[x.cpu().numpy() for x in y_pred_act]
            y_pred_act_np=np.array(y_pred_act_np)
            y_pred_act=torch.tensor(y_pred_act_np).to(device)
            y_pred_act_arr=torch.argmax(y_pred_act, axis=1).to(device)
            print(y_pred_act_arr)
            
            y_onehot_np=[x.cpu().numpy() for x in y_onehot]
            y_onehot_np=np.array(y_onehot_np)
            y_onehot=torch.tensor(y_onehot_np).to(device)
            y_onehot_arr= torch.argmax(y_onehot,axis=1).to(device)
            print("Ground truth ",y_onehot_arr)
            
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            
            
            
            
            auc_metric(y_pred_act, y_onehot)
            f1=f1_metric(y_pred_act_arr, y_onehot_arr)
            if args.opt=="auc":
                result = auc_metric.aggregate()
            elif args.opt=="acc":
                result=acc_metric
            else:
                result=f1
                
            
            
            metric_values.append(result)
            # print("y_pred ",y_pred)
            # print("y", y)
            
            # print("Acc value",acc_value)
            # break
            
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    data_dir, "best_metric_model"+date.today().isoformat()+args.model+args.opt+args.method))
                print("saved new best metric model")
            if args.opt=="auc":    
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
            elif args.opt=="acc":
                print(
                    f"current epoch: {epoch + 1} current AUC: {auc_metric.aggregate():.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best accuracy: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
            else:
                print(
                    f"current epoch: {epoch + 1} current AUC: {auc_metric.aggregate():.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" current F1 : {f1:.4f}"
                    f" best F1 Score: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                
            auc_metric.reset()
            del y_pred_act, y_onehot

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")
    
if args.eval_only==1:
    if args.model=="EfficientNetBN":
        model = locals()[args.model]("efficientnet-b0",spatial_dims=2, in_channels=in_ch,
                            num_classes=num_class,pretrained=False).to(device)
    else:
        model = locals()[args.model](spatial_dims=2, in_channels=in_ch,
                            out_channels=num_class,pretrained=False).to(device)
    model=torch.nn.DataParallel(model)  
    model.load_state_dict(torch.load('./'+args.eval_model))
     
    
else:
    model.load_state_dict(torch.load(
        os.path.join(data_dir, "best_metric_model"+date.today().isoformat()+args.model+args.opt+".pth")))
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        print(model(test_images))
        pred = model(test_images).argmax(dim=1)
                                      
        
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())
        
y_pred=np.convolve(np.array(y_pred),np.ones(5),'same')
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
    




                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
