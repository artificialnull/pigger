import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", required=True, help="video path")
args = vars(ap.parse_args())
filename = args["f"]

table = {}

for line in open(filename).read().split('\n'):
    if len(line) != 0:
        mm, fc = line.split(', ')
        mm = round(float(mm))
        fc = float(fc)
        table[mm] = fc

if len(table) == 0:
    print("empty!")
else:
    newtable = {}
    last = 0
    for i in range(0, max(table.keys()) + 1, 2):
        if i in table.keys():
            newtable[i] = table[i]
        elif i - 1 in table.keys():
            newtable[i] = table[i - 1]
        elif i + 1 in table.keys():
            newtable[i] = table[i + 1]
        else:
            newtable[i] = last
        last = newtable[i]

    for i in range(0, max(newtable.keys()) + 1):
        if i in newtable.keys():
            print("%d, %.3f" % (i, newtable[i]))

