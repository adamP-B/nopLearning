import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

def load_data(name: str, training: bool):
    if name=="CLEVR":
        dir = "/datasets/clevr/CLEVR_v1.0/images/"
        if training:
            dir += "train"
        else:
            dir += "test"
    else:
        raise NameError("Dataset %s not known" % name)
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(64),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])
    data = datasets.ImageFolder(dir, transform=transform)
    return data
