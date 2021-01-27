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

class NoPLearning:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        description = "Train Naming of Parts Learner"
        parser = argparse.ArgumentParser(prog="noLearning",
                                         description=description)
        parser.add_argument("--batch_size", metavar="N", type=int,
                            help="Batch Size", default=256)
        parser.add_argument("--replicas", metavar="N", type=int,
                            help="Number of replicas", default=8)
        parser.add_argument("-e", "--no_epochs", metavar="N", type=int,
                            help="Number of epochs", default=20)
        parser.add_argument("-c", "--no_categories", metavar="N", type=int,
                            help="Number of categories", default=20)
        parser.add_argument("-lr", "--learning_rate", metavar="F", type=float,
                            help="Learning rate", default=0.01)
        parser.add_argument("-w", "--num-workers", metavar="N", type=int,
                            help="Number of workers processing data",
                            default=8)
        parser.add_argument("--dataset", metavar="SetName", type=str,
                            help="Name of dataset", default="CLEVR")
        parser.add_argument('--version', action='version', version=__VERSION__)
        parser.add_argument('--disable-cuda', action='store_true',
                            help='Disable CUDA')
        parser.add_argument('--show-param', action="store_true", default=False)
        self.args = parser.parse_args(sys_argv)
        self.args.device = None
        self.args.use_cuda = None
        if not self.args.disable_cuda and torch.cuda.is_available():
            self.args.device = 'cuda'
            self.args.use_cuda = True
        else:
            self.args.device = 'cpu'
            self.args.use_cuda = False
        print(self.args)
        
    def main(self):
        # housekeeping
        torch.autograd.set_detect_anomaly(True)
        record = {}
        start_time = time.time();
        timeStr = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
        record["start_time"] = timeStr
        record["version"] = type(self).__name__+"."+__VERSION__
        if self.args.use_cuda:
            record["no_GPU"] = torch.cuda.device_count()
        else:
            record["no_GPU"] = 0
        # Setup and train network
        
        net = NoPNet.NoPNet(vars(self.args))

        if self.args.show_param:
            net.printParameters()
            exit(0)
        
        net.dataset(self.args.dataset, record)
        net.train(self.args.no_epochs, record)
        
        record["training_time"] = time.time()-start_time

        # Run tests

        try:
            with torch.no_grad():
                net.examples(record)
        except Exception as e:
            record["tests completion"] = str(e)
        else:
            record["tests completion"] = "Finished"


        # record results
        record["args"] = vars(self.args)
        db = TinyDB("../data/runs/"+type(self).__name__+"_data.json")
        db.insert(record)

if __name__ == "__main__":
    NoPLearning().main()
