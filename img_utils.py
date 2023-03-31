import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
def get_transforms(img_sz,fliph=True,flipv=True,rotate90=True,HSV=True,blur=True,CLAHE=True):
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
        'val': A.Compose([A.CLAHE((2,2),p=1),A.Resize(img_sz,img_sz),A.Normalize(),ToTensorV2()]),
        'onnx_pred':A.Compose([A.CLAHE((2,2),p=1),A.Resize(img_sz,img_sz),A.Normalize()])}
    return data_transforms
    
def preprocess(img_pth,img_sz):
    test_transform=get_transforms(img_sz)['onnx_pred']
    img=cv2.imread(img_pth)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_preprocessed=test_transform(image=img)['image']
    img_preprocessed=np.moveaxis(img_preprocessed, -1, 0)
    img_preprocessed=np.expand_dims(img_preprocessed,axis=0)
    return img_preprocessed