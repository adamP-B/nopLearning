import argparse
from tinydb import TinyDB, Query
from datetime import datetime

db = TinyDB("db.json")


parser = argparse.ArgumentParser(description="Test Arguments")

parser.add_argument("file", metavar="Filename", type=str,
                    help="file to be read")

parser.add_argument("-arg", metavar="N", type=int,
                   help="an argument", default=6)

args = parser.parse_args()

data = {}
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")

data['timestamp'] = timestampStr
data['arglist'] = vars(args)

db.insert(data)
