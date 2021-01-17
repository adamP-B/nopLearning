from tinydb import TinyDB, Query
import pprint
from collections import namedtuple

db = TinyDB("../data/runs/NoPLearning_data.json")
pp = pprint.PrettyPrinter(indent=4)

class Metric(namedtuple('Metric', ['epoch', "kl1", "kl2"])):
    def __str__(self):
        fmt = "[{epoch}] kl1 = {kl1:0.3g}, kl2 = {kl2:0.3g}"
        return fmt.format(**self._asdict())


for run in db:
    final_results = Metric(**run["results"][-1])
    print(run["start_time"], final_results)


# pp.pprint(result)
