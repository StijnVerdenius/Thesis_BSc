import os
for file in os.listdir("."):
    res = ""
    for line in open(file, "r"):
        res = res+line.replace(",", ",")
    f = open(file, "w")
    f.write(res)

