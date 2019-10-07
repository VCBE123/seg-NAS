" for the dataset preprocess transform"
import torchvision.transforms as tvF


def get_trans():
    "get preprocess for data follice"

    traintrans = tvF.Compose(
        [tvF.ToTensor()]
    )
    testtrans = tvF.Compose([
        [tvF.ToTensor()]
    ])
    return traintrans, testtrans
