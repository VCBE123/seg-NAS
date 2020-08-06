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

    def __init__(self, txt, transform=None, return_path=False):
        """
        txt: the path of the txt file
        """
        self.txt = txt
        self.lines = open(self.txt, 'r').readlines()
        self.images = [line.split(' ')[0].strip() for line in self.lines]
        self.labels = [line.split(' ')[1].strip() for line in self.lines]
        self.trans = transform
        # print(len(self.images))
        self.return_path=return_path
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        label = cv2.imread(self.labels[index], 0)
        if self.trans:
            image, label = self.trans(image, label)
        if self.return_path:
            return image, label, self.images[index],self.labels[index]
        return image, label


def get_follicle(batch_size, workers=4, train_aug=False, test_aug=False,return_path=False):
    "return trainloader,testloader"
    train_trans = ImgAugTrans(crop_size=384, aug=train_aug)
    test_trans = ImgAugTrans(crop_size=384, aug=test_aug)

    trainset = FollicleDataset(
        '/data/lir/follicle/train.txt', train_trans)
    testset = FollicleDataset('/data/lir/follicle/eval.txt', test_trans,return_path=return_path)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=workers, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                            num_workers=workers, pin_memory=True, drop_last=False)
    return trainloader, testloader


if __name__ == "__main__":
    pass
