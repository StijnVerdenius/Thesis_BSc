import matplotlib.pyplot as plt
import numpy as np
import time
from dateutil import parser
from matplotlib import rc


f = open("tijd_experiment_2.csv", "r")
# f = open("time2.csv", "r")

data = []

vorige = parser.parse("Fri Jun 8 17:27:32 2018")

for line in f:
    x,y = line.split(",")
    x = int(x)*0.05-0.05
    datum = parser.parse(y)
    verschil = datum-vorige
    data.append([x,verschil.seconds])
    vorige = datum

data = np.array(data[1:])

plt.axis((0,1.0,0,1900))
plt.plot(data[:,0], data[:,1])
plt.xlabel('Entropy of instance ')
plt.ylabel('Time per epoch (s)')
plt.title('Progression of time complexity with entropy')
plt.show()