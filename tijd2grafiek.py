import matplotlib.pyplot as plt
import numpy as np
import time
from dateutil import parser

dir = "workresults/score/"


filenames = ["20_f0_f1_{}".format(x) for x in ["basel", "adap", "fixed"]]

# filenames = ["testding.csv", "testding2.csv"]
label = ["No Curriculum", "Adaptive Curriculum", "Fixed Curriculum"]

data = []
for i, filename in enumerate(filenames):

    f = open(dir+filename+".csv", "r")

    data = []

    eersteWall = 0

    for line in f:
        wall, step, value = line.split(",")

        if (not step == "Step"):

            if(eersteWall == 0):
                eersteWall = float(wall)

            step = 100*int(step)/250000.0

            data.append([float(wall)-eersteWall, int(step), float(value)])

    data = np.array(data)

    plt.plot(data[:,0], data[:,2], label=label[i])

plt.xticks([])

x1,x2,y1,y2 = plt.axis()
# plt.axis((x1,x2,3.8,3.9))
plt.xlabel('Time relative')
plt.ylabel('Tour-length')
plt.legend(loc="upper right")
plt.title('Progression of tour-length on validationset over time')
plt.show()