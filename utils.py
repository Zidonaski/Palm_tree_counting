import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
IMG_SZ=416
class TreeCountingDataset(Dataset):
  def __init__(self, imgdir ,csv_data,transform=None,dataset_type="train"):
        self.imgdir=imgdir
        self.data = csv_data
        self.transform=transform 
        self.dataset_type=dataset_type      
  def __len__(self):
        return len(self.data)
    
  def __getitem__(self, idx):
       
        imagename = self.data.iloc[idx].ImageId
        img=cv2.imread(join(self.imgdir,imagename))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
          img=self.transform(image=img)['image']

        if  self.dataset_type=="train":
          label= self.data.iloc[idx].Target
          return img, label
        else :
          return img

def get_train_data(imgdir,csv_files_pth,batch_size,transforms=None):
    train_file_pth=join(csv_files_pth,'Train.csv')
    train_df=pd.read_csv(train_file_pth)

    train_dataset=TreeCountingDataset(imgdir,train_df,transforms,"train")
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size,shuffle=True,drop_last=True)
    return train_dataset,train_dataloader

def get_val_data(imgdir,csv_files_pth,batch_size,transforms=None):
    val_file_pth=join(csv_files_pth,'Val.csv')
    val_df=pd.read_csv(val_file_pth)

    val_dataset=TreeCountingDataset(imgdir,val_df,transforms,"train")
    val_dataloader=DataLoader(val_dataset, batch_size=batch_size,shuffle=True,drop_last=True)
    return val_dataset,val_dataloader

def get_test_data(imgdir,csv_files_pth,batch_size,transforms=None):
    test_file_pth=join(csv_files_pth,'Test.csv')
    test_df=pd.read_csv(test_file_pth)

    test_dataset=TreeCountingDataset(imgdir,test_df,transforms,"train")
    test_dataloader=DataLoader(test_dataset, batch_size=batch_size,shuffle=True,drop_last=True)
    return test_dataset,test_dataloader

def get_transforms(img_sz=IMG_SZ,fliph=True,flipv=True,rotate90=True,HSV=True,blur=True,CLAHE=True):
    transforms=[]
    if CLAHE:
        transforms.append(A.CLAHE((2,2),p=1))
    if fliph:
        transforms.append(A.HorizontalFlip(p=0.5))
    if flipv:
        transforms.append(A.VerticalFlip(p=0.5))
    if rotate90:
        transforms.append(A.RandomRotate90(p=0.5))
    if HSV:
        transforms.append(A.HueSaturationValue(p=0.5))
    if blur:
        transforms.append(A.GaussianBlur(p=0.5))

    transforms.append(A.Resize(img_sz,img_sz))
    transforms.append(A.Normalize())
    transforms.append(ToTensorV2)
    data_transforms = {'train': A.Compose(transforms),
        'val': A.Compose([A.CLAHE((2,2),p=1),A.Resize(img_sz,img_sz),A.Normalize(),ToTensorV2()])}
    return data_transforms
def seed_all():
    torch.manual_seed(77)
    random.seed(77)
    np.random.seed(77)