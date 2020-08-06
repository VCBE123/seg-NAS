import cv2
import numpy as np
import matplotlib as plt
import glob

base='logs/base_line/*.png'
images=glob.glob(base)
images.sort()
for image in images:
    print(image)

    base_image=cv2.imread(image)

    ori,pro=np.split(base_image,2,axis=1)

    fig_image=cv2.imread(image.replace('base_line','fig'))

    _,label,base=np.split(fig_image,3,axis=1)

    #####
    mask=label[:,:,1].copy()
    follicle=mask.copy()
    follicle[follicle==255]=0
    follicle[follicle==128]=255
    mask[mask==128]=0
    label[:,:,2]=0
    label[:,:,1]=mask
    label[:,:,0]=follicle


    ###
    mask=base[:,:,1].copy()
    follicle=mask.copy()
    follicle[follicle==255]=0
    follicle[follicle==128]=255
    mask[mask==128]=0
    base[:,:,2]=0
    base[:,:,1]=mask
    base[:,:,0]=follicle
    ###
   
    mask=pro[:,:,1].copy()
    follicle=mask.copy()
    follicle[follicle==255]=0
    follicle[follicle==128]=255
    mask[mask==128]=0
    pro[:,:,2]=0
    pro[:,:,1]=mask
    pro[:,:,0]=follicle

    result=np.concatenate([ori,label,base,pro],1)
    # result=cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(image.replace('base_line','plot'),result)