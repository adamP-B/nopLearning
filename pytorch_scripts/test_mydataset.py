import matplotlib.pyplot as plt
from mydataset import load_data
# Ignore warnings
import warnings


dataset = load_data("CLEVR", True)

print("Number of images = %r" % len(dataset))


plt.imshow(dataset[0])
