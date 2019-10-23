" for the dataset preprocess transform"
import numpy as np
import torchvision.transforms as tvF
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

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
        if aug:
            self.aug = iaa.Sequential([
                iaa.Resize({"height": crop_size, "width": crop_size}),
                iaa.Dropout([0.05, 0.2]),
                iaa.Sharpen((0.0, 1.0)),
                iaa.Affine(rotate=(-20, 20)),
                iaa.ElasticTransformation(alpha=50, sigma=5)])
        else:
            self.aug = iaa.Sequential(
                [iaa.Resize({"height": crop_size, "width": crop_size}), ])
        self.normalize = tvF.Compose([tvF.ToTensor(), NORMALIZE])
        self.totensor = tvF.Compose([tvF.ToTensor()])
        self.num_classes = num_classes

    def __call__(self, image, mask):
        image = np.asarray(image)
        mask = np.asarray(mask, dtype=np.int32)
        mask[mask  <100] = 0
        mask[mask > 200] = 2
        mask[mask>2] =1

        # imgaug
        mask = SegmentationMapsOnImage(mask, shape=image.shape)
        image, aug_mask = self.aug(image=image, segmentation_maps=mask)
        mask = aug_mask.get_arr()
        #"one-hot encode"
        mask = np.eye(self.num_classes)[mask]

        image_norm = self.normalize(image)
        mask = self.totensor(mask).float()
        return image_norm, mask
