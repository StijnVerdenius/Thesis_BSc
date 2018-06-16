import matplotlib.pyplot as plt
import numpy as np
import time
from dateutil import parser


filenames = ["testding.csv"]*2
label = ["ananas"]*2


for i, filename in enumerate(filenames):

    f = open(filename, "r")

    data = []

    for line in f:
        wall, step, value = line.split(",")

        if (not step == "Step"):

            step = 100*int(step)/250000.0

            data.append([float(wall), int(step), float(value)])

    data = np.array(data)+i

    plt.plot(data[:,1], data[:,2], label=label[i])

plt.axis((0,55,0,1900))
plt.xlabel('Epoch')
plt.ylabel('Tour-length')
plt.legend(loc="upper right")
plt.title('Progression of tour-length on validationset')
plt.show()
