" for the dataset preprocess transform"
import torchvision.transforms as tvF
from imgaug import augmenters as iaa
import numpy as np


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

    def __init__(self, input_size, crop_size, num_classes=3):
        self.aug = iaa.Sequential([
            iaa.Scale({"height": input_size, "width": input_size}),
            iaa.Crop(px=input_size-crop_size)])
        self.normalize = tvF.Compose([tvF.ToTensor(), NORMALIZE])
        self.totensor = tvF.Compose([tvF.ToTensor()])
        self.num_classes = num_classes

    def __call__(self, image, mask):
        seg_det = self.aug.to_deterministic()
        image = np.asarray(image)
        mask = np.asarray(mask)
        "one-hot encode"
        print(mask.shape)
        print(np.unique(mask))
        mask[mask == 128] = 1
        mask[mask == 255] = 2
        mask = np.eye(self.num_classes)[mask]
        aug_image = seg_det.augment_image(image)
        aug_mask = seg_det.augment_image(mask)
        aug_norm = self.normalize(aug_image)
        aug_mask = self.totensor(mask)
        return aug_norm, aug_mask
