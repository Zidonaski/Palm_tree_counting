from torch import nn,optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import *
from model import CNN


def train(trainloader,valloader , epochs , model,lr,unfreeze_ep=2):
    cuda=torch.cuda.is_available()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn=nn.MSELoss()
    model=model.cuda() if cuda else model
    train_losses=[]
    val_losses=[]
    model.freeze()
    for e in range(epochs):
        if e+1 ==unfreeze_ep :
            model.unfreeze()
            print('layer unfreezed')
        train_samples=0
        train_cumalitive_loss=0
        model.train()
        for i, data in enumerate(trainloader):
            images,labels=data
            (images,labels)= (images.cuda(),labels.float().cuda()) if cuda else (images,labels.float())
            predictions=model(images)
            predictions=predictions.view(labels.shape)
            optimizer.zero_grad()
            loss=loss_fn(predictions,labels)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            train_samples=train_samples+images.shape[0]
            train_cumalitive_loss+=loss.item()*images.shape[0]
        
        train_epoch_loss=np.sqrt( train_cumalitive_loss/train_samples )
        train_losses.append(train_epoch_loss)
        model.eval()

        val_cumalitive_loss=0
        val_samples=0
        for i, data in enumerate(valloader):
            images,labels=data
            images,labels=(images.cuda(),labels.float().cuda()) if cuda else (images,labels.float())
            predictions=model(images)
            predictions=predictions.view(labels.shape)
            loss=loss_fn(predictions,labels)
            val_cumalitive_loss+=loss.item()*images.shape[0]
            val_samples=val_samples+images.shape[0]
        val_epoch_loss=np.sqrt( val_cumalitive_loss/len(val_samples) )
        val_losses.append(val_epoch_loss)
        print("Epoch : %2d Train loss = %4.2f Test loss = %4.2f "%(e+1,train_epoch_loss,val_epoch_loss))
        if val_epoch_loss==min(val_losses):
            torch.save(model.state_dict(),"models/best_model.pt")
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.legend()
    plt.pause(0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='Train.csv', help='csv file with train images')
    parser.add_argument('--val', type=str, default='Val.csv', help='csv file with validation images')
    parser.add_argument('--imgdir',type=str,default='images/',help='image directory')
    parser.add_argument('--epochs', type=int,default=20,help='number of epochs to train')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate to be used')
    parser.add_argument('--batchsz',type=int,default=16,help='batch size')
    parser.add_argument('--imgsz',type=int,default=416,help="image size")
    
    opt = parser.parse_args()
    seed_all()
    transfroms=get_transforms(opt.imgsz)
    train_dataset,train_dataloader=get_train_data(opt.imgdir,opt.train,opt.batchsz,transfroms)
    val_dataset,val_dataloader=get_val_data(opt.imgdir,opt.val,opt.batchsz,transfroms)
    Treeconting_model=CNN()
    train(train_dataloader,val_dataloader,opt.epochs,Treeconting_model,opt.lr)

