import sys
from datetime import datetime
import time
import argparse
import torch
from tinydb import TinyDB, Query
from gitversion import rewritable_git_version
__VERSION__ = rewritable_git_version(__file__)
import NoPNet
import warnings
warnings.filterwarnings("ignore") 
import results

class NoPTest:
    def __init__(self, sys_argv=None):
        _, self.record = results.Results().select()
        self.args = self.record["args"]

        if not self.args['disable_cuda'] and torch.cuda.is_available():
            self.args['device'] = 'cuda'
            self.args['use_cuda'] = True
        else:
            self.args['device'] = 'cpu'
            self.args['use_cuda'] = False
        print(self.args)
        
    def main(self):
        # housekeeping
        record = {}
        
        net = NoPNet.NoPNet(self.args)
        net.load()
        net.dataset(self.args["dataset"], record)
        

        # Run tests

        try:
            net.examples(record)
        except Exception as e:
            print("Tests are broken")
            print(e)
        else:
            print("Tests run")


        # record results

if __name__ == "__main__":
    NoPTest().main()
