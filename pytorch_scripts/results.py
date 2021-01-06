from tinydb import TinyDB, Query
import pprint

db = TinyDB("../data/runs/NoPLearning_data.json")
pp = pprint.PrettyPrinter(indent=4)

for result in db:
    pp.pprint(result)
