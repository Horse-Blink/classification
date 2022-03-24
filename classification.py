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
from sklearn.metrics import classification_report
from datetime import date, datetime
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, DenseNet264
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

# print_config()
torch.multiprocessing.set_sharing_strategy("file_system")

parser=argparse.ArgumentParser()

parser.add_argument("--batch_size", default=1, type=int,help="number of images to process for each step of gradient descent")
parser.add_argument("--model",default="DenseNet121",type=str, help="DenseNet121 or DenseNet264")
parser.add_argument("--load_save",default=0, type=int,help="load saved weights from previous training")
parser.add_argument("--epochs", default=100, type=int, help="number of epochs to run")
parser.add_argument("--opt",default="acc",type=str, help= "Optimisation metric to use- 'auc' or 'acc'")
parser.add_argument("--comb_ch",default=1,type=int, help="train with all 3 channels or combine them into 1")

args=parser.parse_args()
print(' '.join(sys.argv))

data_dir = './'
print(data_dir)

set_determinism(seed=0)

class_names = ["Blinking","Eyes Open"]

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
# np.random.shuffle(indices)

# test_split = int(test_frac * length)
# val_split = int(val_frac * length) + test_split
test_indices = [i for i,item in enumerate(image_files_list) if re.search("GH010022",item)]#indices[:176]
val_indices = [i for i,item in enumerate(image_files_list) if re.search("GH010007",item)]#indices[test_split:val_split]
train_indices = [i for i in indices if i not in val_indices and i not in test_indices]



train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

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
        
        
        if args.comb_ch==1:     
            image=image[0,:,:]*0.114+image[1,:,:]*0.587+image[2,:,:]*0.299
            image=cv2.resize(np.array(image),(720,960),interpolation=cv2.INTER_AREA)
                
            image=image[None,:,:]
            
        return image, self.labels[index]


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
model = locals()[args.model](spatial_dims=2, in_channels=in_ch,
                    out_channels=num_class).to(device)

with torch.cuda.amp.autocast():
    summary(model,(in_ch,720,960))

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
max_epochs = args.epochs
val_interval = 1
auc_metric = ROCAUCMetric()
if args.load_save==1:
    model.load_state_dict(torch.load(
        os.path.join(data_dir, "best_metric_model"+args.model+args.opt+".pth")))
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
                f"train_loss: {loss.item():.4f}")
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
            # print("y onehot",y_onehot)
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)] #softmax
            
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            
            
            # print("y pred act",y_pred_act)
            auc_metric(y_pred_act, y_onehot)
            if args.opt=="auc":
                result = auc_metric.aggregate()
            else:
                result=acc_metric
                
            
            
            metric_values.append(result)
            # print("y_pred ",y_pred)
            # print("y", y)
            
            # print("Acc value",acc_value)
            # break
            
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    data_dir, "best_metric_model"+date.today().isoformat()+args.model+args.opt+".pth"))
                print("saved new best metric model")
            if args.opt=="auc":    
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
            else:
             print(
                    f"current epoch: {epoch + 1} current AUC: {auc_metric.aggregate():.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best accuracy: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
            auc_metric.reset()
            del y_pred_act, y_onehot

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")
    
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
        pred = model(test_images).argmax(dim=1)
        
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())
        
y_pred=np.convolve(np.array(y_pred),np.ones(5),'same')
y_pred=(y_pred>0.5).astype(int)  
        
print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4))