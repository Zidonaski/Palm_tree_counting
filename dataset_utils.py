import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
from img_utils import *
import cv2
import random

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

def seed_all():
    torch.manual_seed(77)
    random.seed(77)
    np.random.seed(77)