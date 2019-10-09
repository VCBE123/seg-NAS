"""
load follice dataset in /data/follice
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from .img_preprocess import ImgAugTrans


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
        self.images = [
            '/data/lir'+line.split('..')[1].strip() for line in self.lines]
        self.labels = [line.replace('image', 'label') for line in self.images]
        self.trans = transform
        print(len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        label = cv2.imread(self.labels[index], 0)
        if self.trans:
            image, label = self.trans(image, label)
        return image, label


def get_follicle(args):
    "return trainloader,testloader"
    train_trans = ImgAugTrans(crop_size=384)
    test_trans = ImgAugTrans(crop_size=384)

    trainset = FollicleDataset('/data/follicle/train_image.txt', train_trans)
    testset = FollicleDataset('/data/follicle/eval_image.txt', test_trans)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.workers, pin_memory=True)
    testloader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    return trainloader, testloader


if __name__ == "__main__":
    DATASET = FollicleDataset('/data/follicle/train_image.txt')
    # image,label=DATASET.__getitem__(0)
    # print(label.size)
