import os
import numpy as np
# for file in os.listdir("."):
#     if ("addaptive" in file and "time" in file):
#         res = ""
#         for line in open(file, "r"):
#             elems = line.split(",")
#             if (len(elems) == 116):
#                 res = res + line
#         f = open(file, "w")
#         f.write(res)


f = open("auto_08052018_score_in_time_validationset.csv", "r")

res = ""

for line in f:
    elems = line.split(",")
    elems = [float(k) for k in elems]
    newLine = [elems[0]]
    if (len(elems) > 117):
        relevant = elems[1:(24*5)+1]
        for i in range(int(len(relevant)/8)):
            newLine.append(np.mean(relevant[8*i:(i+1)*8]))
        newLine = newLine + elems[(24*5)+1:]
        newLine = str(newLine).replace("[", "").replace("]", "") + "\n"
        res = res + newLine
    else:
        res = res + line



f.close()

g = open("auto_08052018_score_in_time_validationset.csv", "w")

g.write(res)