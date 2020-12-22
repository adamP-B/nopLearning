import os
import pickle

def makeDataList(dirRoot, fn):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dirRoot):
        for f in filenames:
            files.append(os.path.join(dirpath, f))
    
    pickle.dump(files, open(fn+".p", "wb"))



if __name__ == '__main__':
    makeDataList('../../MultiDigitMNIST/dataset/five_mnist/train/', '../data/five_mnist')
