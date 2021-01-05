import sys
from datetime import datetime
import time
import argparse
from tinydb import TinyDB, Query

class NoPLearning:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        description = "Train Naming of Parts Learner"
        parser = argparse.ArgumentParser(prog="noLearning",
                                         description=description)
        parser.add_argument("-batch_size", metavar="N", type=int,
                   help="Batch Size", default=6)
        parser.add_argument("-learning_rate", metavar="N", type=float,
                            help="Learning Rate", default=0.01)
        self.my_args = parser.parse_args(sys_argv)
        self.batch_size = self.my_args.batch_size
        self.learning_rate = self.my_args.learning_rate
        
    def main(self):
        data = {}
        start_time = time.time();
        timeStr = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        data["start_time"] = timeStr
        data["args"] = vars(self.my_args)
        
        

        data["run_time"] = time.time()-start_time
        db = TinyDB("../data/runs/"+type(self).__name__+"_data.json")
        db.insert(data)

if __name__ == "__main__":
    NoPLearning().main()
