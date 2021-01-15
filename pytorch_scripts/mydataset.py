import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from enum import Enum
from PIL import Image

class DataSetType(Enum):
    Train = 0
    Val = 1
    Test = 2

class CLEVR(Dataset):
    """Loads the CLEVR dataset"""

    def __init__(self, datasettype: DataSetType, transform=None):
        """loads a dataset"""
        dirs = ("train/", "val/", "test/")
        dirRoot = "../data/CLEVR/" + dirs[datasettype.value]
        self.transform = transform
        self.files = []
        for (dirpath, dirnames, filenames) in os.walk(dirRoot):
            for f in filenames:
                self.files.append(os.path.join(dirpath,f))
        self.len = len(self.files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_file = self.files[idx]
        image = Image.open(image_file)
#        image = torch.tensor(image, dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        return image[0:3,:,:]
            
def load_data(name: str, datasettype: DataSetType):
    """ General loader """
    
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
         ])

    if name=="CLEVR":
        return CLEVR(datasettype, transform)
    else:
        raise NameError("Dataset %s not known" % name)
    if training:
        dir += "train/"
    else:
        dir += "test/"
    data = datasets.ImageFolder(root=dir, transform=transform)
    return data

