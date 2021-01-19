import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import math
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from mydataset import load_data, DataSetType

from NoPNet import NoPModel


dataset = load_data("CLEVR", DataSetType.Train)

print("Number of images = %r" % len(dataset))


print("Shape = ", dataset[-1].shape)


def multiplot(figures, cols: int=3):
    rows = math.ceil(len(figures)/cols)
    fig, ax = plt.subplots(rows, cols)
    c = 0
    r = 0
    for f in figures:
        ax[r,c].imshow(f.permute(1,2,0))
        c += 1
        if c==cols:
            r += 1
            c = 0
    plt.show()


no_categories = 22
device = torch.device('cpu')

model = NoPModel(no_categories, device)

ds = dataset[0].unsqueeze(0).expand(9,3,64,64)
print(ds.shape)

bb = 0.8*torch.rand(9,4)+0.1
print(bb)
print(bb.shape)
patches = model.spatial(ds, bb)

multiplot(patches)
