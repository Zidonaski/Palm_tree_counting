import torch
from torch import nn
import argparse
from model import CNN
import numpy as np
from utils import get_test_data, get_transforms

def test(testloader, model):
    cuda=torch.cuda.is_available()
    model= model.cuda() if cuda else model
    model.eval()
    loss_fn=nn.MSELoss()
    preds=[]
    
    test_cumalitive_loss=0
    test_samples=0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images,labels=data
            images,labels=(images.cuda(),labels.float().cuda()) if cuda else (images,labels.float())
            predictions=model(images)
            predictions=predictions.view(labels.shape)
            preds.append(predictions.item())
            loss=loss_fn(predictions,labels)
            test_cumalitive_loss+=loss.item()*images.shape[0]
            test_samples=test_samples+images.shape[0]

    test_loss=np.sqrt( test_cumalitive_loss/len(test_samples) )
    print("Test loss = %4.2f "%(test_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='Test.csv', help='csv file with test images')
    parser.add_argument('--imgdir',type=str,default='images/',help='image directory for test')
    parser.add_argument('--weigths',type=str,default='best_model.pt',help="path of the trained model")
    parser.add_argument('--batchsz',type=int,default=16,help='batch size')
    parser.add_argument('--imgsz',type=int,default=416,help="image size")
    opt = parser.parse_args()
    Treecounting_model=CNN()
    Treecounting_model.load_state_dict(torch.load(opt.weigths,map_location=torch.device('cpu')))
    transforms=get_transforms(opt.imgsz)
    test_dataset,test_dataloader=get_test_data(opt.imgdir,opt.test,opt.batchsz,transforms)
    test(test_dataloader,Treecounting_model)
