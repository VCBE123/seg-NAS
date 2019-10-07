"""
load follice dataset in /data/follice
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from .img_preprocess import get_trans


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
        self.images = [line.split(' ')[0] for line in self.lines]
        self.labels = [line.split(' ')[1] for line in self.lines]
        self.trans = transform
        print(len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        label = cv2.imread(self.labels[index])
        if self.trans:
            image, label = self.trans
        return image, label


def get_follicle(args):
    "return trainloader,testloader"
    train_trans, test_trans = get_trans()

    trainset = FollicleDataset('/data/follicle/train_all.txt', train_trans)
    testset = FollicleDataset('/data/follicle/eval.txt', test_trans)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.workers, pin_memory=True)
    testloader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    return trainloader, testloader


if __name__ == "__main__":
    DATASET = FollicleDataset('/data/follicle/train_all.txt')
