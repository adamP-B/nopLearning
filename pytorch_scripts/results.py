#!/home/apb/anaconda3/bin/python

import os
from tinydb import TinyDB, Query
import pprint
from collections import namedtuple
from ask import Ask

pp = pprint.PrettyPrinter(indent=4)


class Results:
    """Extract and Process Results"""

    def __init__(self):
        """Setup"""
        self.db = TinyDB("../data/runs/NoPLearning_data.json")
        self.record = Query()
        self.ask = Ask()

    
    def overview(self):
        for run in self.db:
            results = run["training_data"]['test'][-1]
            pp.pprint({"id":run.doc_id, "start":run["start_time"], "result":results})

    def select(self):
        self.overview()
        while True:
            n = int(self.ask("What input do you want to see", "select", 0))
            if self.db.contains(doc_id=n):
                break
            else:
                print("unknown id")
        return n, self.db.get(doc_id=n)

    def show(self):
        n, result = self.select()
        print("Showing results for id {:d}".format(n))
        pp.pprint(result)

    def delete(self):
        n, result = self.select()
        pp.pprint(result)
        ans = input("Are you sure you want to delete record [N/y]? ")
        if ans == "y":
            os.remove(result['args']["weightFile"])
            self.db.delete(doc_id=n)

    def garbage_collection(self):
        for root, dirs, files in os.walk("../data/runs/weights"):
            print(files)
            for f in files:
                if not self.db.contains(self.record.args.weightFile == f):
                    print("removing", f)
                    os.remove(root+"/"+f)
                else:
                    print("keeping", f)



    def selectAction(self):
        actions = {"overview": self.overview,
                   "show": self.show,
                   "delete": self.delete,
                   "garbage collection": self.garbage_collection,
                   "quit": exit
                   }
        while True:
            actionList = []
            for i, action in enumerate(actions.keys()):
                actionList.append(action)
                print(i, action)
            n = int(self.ask("What action do you want to take", "action", 0))
            if n>=0 and n<len(actionList):
                break
        actions[actionList[n]]()

        

    def loop(self):
        while True:
            self.selectAction()

if __name__ == "__main__":
    Results().loop()

