" for the dataset preprocess transform"
import numpy as np
import torchvision.transforms as tvF
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.pyplot as plt

ia.seed(109)


def get_trans():
    "get preprocess for data follice"

    traintrans = tvF.Compose([tvF.ToTensor()]
                             )
    testtrans = tvF.Compose([tvF.ToTensor()])
    return traintrans, testtrans


NORMALIZE = tvF.Normalize(mean=[0.275, 0.278, 0.284],
                          std=[0.170, 0.171, 0.173])


class ImgAugTrans:
    "augment image and the mask"

    def __init__(self, crop_size=384, num_classes=3, aug=True):
        self.aug = iaa.Sequential([
            iaa.Resize({"height": crop_size, "width": crop_size}),
            iaa.Dropout([0.05, 0.2]),
            iaa.Sharpen((0.0, 1.0)),
            iaa.Affine(rotate=(-20, 20)),
            iaa.ElasticTransformation(alpha=50, sigma=5)])

        self.normalize = tvF.Compose([tvF.ToTensor(), NORMALIZE])
        self.totensor = tvF.Compose([tvF.ToTensor()])
        self.num_classes = num_classes
        self.aug=aug

    def __call__(self, image, mask):
        image = np.asarray(image)
        mask = np.asarray(mask, dtype=np.int32)

        # imgaug
        if self.aug:
            mask = SegmentationMapsOnImage(mask, shape=image.shape)
            image, aug_mask = self.aug(image=image, segmentation_maps=mask)

        #"one-hot encode"
        # print(mask.shape)
        # print(np.unique(mask))
            mask = aug_mask.get_arr()
        mask = np.eye(self.num_classes)[mask]
        

        #
        # aug_image=aug_image.cpu.numpy()
        # plt.imshow(aug_image)
        # plt.show()
        
        image_norm = self.normalize(image)
        mask = self.totensor(mask).float()
        return image_norm, mask
