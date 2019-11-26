"""
load follice dataset in /data/follice
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from .img_preprocess import ImgAugTrans
import numpy as np


class FollicleDataset(Dataset):

    """
    load follice dataset
    """

    def __init__(self, txt, transform=None):
        """
        txt: the path of the txt file
        """
        self.txt = txt
        self.lines = open(self.txt, 'r').readlines()
        self.images = [line.split(' ')[0].strip() for line in self.lines]
        self.labels = [line.split(' ')[1].strip() for line in self.lines]
        self.trans = transform
        # print(len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        label = cv2.imread(self.labels[index], 0)
        if self.trans:
            image, label = self.trans(image, label)
        return image, label


def get_follicle(batch_size, workers, train_aug=False, test_aug=False):
    "return trainloader,testloader"
    train_trans = ImgAugTrans(crop_size=384, aug=train_aug)
    test_trans = ImgAugTrans(crop_size=384, aug=test_aug)

    trainset = FollicleDataset(
        '/data/lir/follicle/train_pain.txt', train_trans)
    testset = FollicleDataset('/data/lir/follicle/eval.txt', test_trans)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=workers, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                            num_workers=workers, pin_memory=True, drop_last=False)
    return trainloader, testloader


if __name__ == "__main__":
    pass
