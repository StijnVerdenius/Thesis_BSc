import matplotlib.pyplot as plt
import numpy as np
import time
from dateutil import parser

dir = "workresults/score/"
# filenames = ["testding.csv" ]
filenames = ["20_f0_f1_{}".format(x) for x in ["basel", "adap", "fixed"]]
# filenames = ["20_f0_f1_{}".format("basel"), "20_f0_xx_fixed", "20_xx_f1_fixed"]

label = ["No Curriculum", "Adaptive Curriculum", "Fixed Curriculum"]
# label = ["No Curriculum", "Fixed Curriculum (size only)", "Fixed Curriculum (entropy only)"]

for i, filename in enumerate(filenames):

    f = open(dir+filename+".csv", "r")

    data = []

    for line in f:
        wall, step, value = line.split(",")

        if (not step == "Step"):

            step = 100*int(step)/250000.0

            data.append([float(wall), int(step), float(value)])

    data = np.array(data)

    plt.plot(data[:,1], data[:,2], label=label[i])

plt.axis((0,100,3.8,4.0))
plt.xlabel('Epoch')
plt.ylabel('Tour-length')
plt.legend(loc="upper right")
plt.title('Progression of tour-length on validationset')
plt.show()
