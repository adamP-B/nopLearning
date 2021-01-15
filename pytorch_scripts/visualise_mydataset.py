import matplotlib.pyplot as plt
from mydataset import load_data, DataSetType
from torch.utils.data import DataLoader
import math
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


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

figs = [dataset[i] for i in range(0,9)]
multiplot(figs)

