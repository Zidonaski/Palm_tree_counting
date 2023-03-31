import albumentations as A
import cv2
import numpy as np    
def preprocess(img_pth,img_sz):
    onnx_transform=A.Compose([A.CLAHE((2,2),p=1),A.Resize(img_sz,img_sz),A.Normalize()])
    img=cv2.imread(img_pth)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_preprocessed=onnx_transform(image=img)['image']
    img_preprocessed=np.moveaxis(img_preprocessed, -1, 0)
    img_preprocessed=np.expand_dims(img_preprocessed,axis=0)
    return img_preprocessed