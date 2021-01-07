import sys
from datetime import datetime
import time
import argparse
import torch
from tinydb import TinyDB, Query
from gitversion import rewritable_git_version
__VERSION__ = rewritable_git_version(__file__)
import NoPNet

class NoPLearning:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        description = "Train Naming of Parts Learner"
        parser = argparse.ArgumentParser(prog="noLearning",
                                         description=description)
        parser.add_argument("-b", "-batch_size", metavar="N", type=int,
                   help="Batch Size", default=16)
        parser.add_argument("-e", "-no_epochs", metavar="N", type=int,
                            help="Number of epochs", default=100)
        parser.add_argument("-c", "-no_categories", metavar="N", type=int,
                            help="Number of categories", default=20)
        parser.add_argument("-lr", "-learning_rate", metavar="F", type=float,
                            help="Learning rate", default=0.01)
        parser.add_argument("-w", "-num-workers", metavar="N", type=int,
                            help="Number of workers processing data",
                            default=4)
        parser.add_argument("-dataset", metavar="SetName", type=str,
                            help="Name of dataset", default="CLEVR")
        parser.add_argument('--version', action='version', version=__VERSION__)
        self.args = parser.parse_args(sys_argv)
        
    def main(self):
        data = {}
        start_time = time.time();
        timeStr = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
        data["start_time"] = timeStr
        data["args"] = vars(self.args)
        data["version"] = type(self).__name__+"."+__VERSION__

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        net = NoPNet.NoPTrain(self.args, self.use_cuda, self.device)
        data["results"] = net.run(self.args.dataset , self.args.no_epochs)
        data["device"] = str(self.device)
        if self.use_cuda:
            data["no_GPU"] = torch.cuda.device_count()
        else:
            data["no_GPU"] = 0
        data["run_time"] = time.time()-start_time
        db = TinyDB("../data/runs/"+type(self).__name__+"_data.json")
        db.insert(data)

if __name__ == "__main__":
    NoPLearning().main()
