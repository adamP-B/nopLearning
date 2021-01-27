# ask.py

import os
import pickle

class Ask:
    """Ask for input and store results"""

    def __init__(self, fn = "./.saveAsk"):
        self.sf = fn
        self.saved = {}
        if (os.path.isfile(self.sf)):
            with open(self.sf, 'rb') as fp:
                self.saved = pickle.load(fp)

    
    def __call__(self, question, key, default=0):
        value = self.saved.get(key, default)
        self.saved[key] = value
        question += " [" + str(value) + "]? "
        ans = input(question) or value
        if self.saved[key] != ans:
            self.saved[key] = ans
            with open(self.sf, 'wb') as fp:
                pickle.dump(self.saved, fp)
        return ans
