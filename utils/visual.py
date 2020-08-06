import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


image_folder='image/*.png'
label_folder='label/*.png'
mask_folder='mask/*.png'

for image in glob.glob(mask_folder):
    img=cv2.imread(image)
    cv2.imwrite(image.split('/')[-1],img*127)


for image in glob.glob(image_folder):
    img=cv2.imread(image)

    label=cv2.imread(image.replace('image','label'),0)

    follicle=label.copy()
    ovary=label.copy()
    ovary[ovary>=1]=1
    follicle[follicle>1]=0

    _,ovary_mask=cv2.threshold(ovary,0,1,cv2.THRESH_BINARY)
    ovary_cont,_=cv2.findContours(ovary_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,ovary_cont,-1,(0,0,255),1)

    _,follicle_mask=cv2.threshold(follicle,0,1,cv2.THRESH_BINARY)
    ovary_cont,_=cv2.findContours(follicle_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,ovary_cont,-1,(0,255,255),1)


    

    cv2.imwrite(image.split('/')[1],img)