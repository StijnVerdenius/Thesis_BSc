import matplotlib.pyplot as plt
import numpy as np

dingen = [

    (0,0,23),(1,2,20)

]


##################

for iii, sort in enumerate([("validationset", 86, "lower right"), ("loss", 750, "upper right")]):

    curs = dingen
    for cur in curs:
        plt.title(str(cur))
        data = []
        for line in open("results/lang_"+str(cur)+"_score_in_time_"+sort[0]+".csv"):
            listLine = [float(i) for i in line.split(",")[1:]]
            data.append(listLine)
        dataNP = np.array(data)
        dataTP = dataNP.T
        dataTPL = dataTP.tolist()
        newdata = []
        for col in dataTPL:
            newdata.append(np.mean(col))

        newdata2 = []
        for yy in range(int(len(newdata)/5)):
            newdata2.append(np.mean(newdata[yy*5:(yy+1)*5]))
        newdata = newdata2

        plt.axis((int(len(newdata)*0.0), len(newdata)-1, 75, sort[1]))

        plt.plot(newdata, label=str(cur))


        if (iii == 0):
            data = []
            for line in open("results/lang_" + str(cur) + "_score_in_end.csv"):
                listLine = [float(i) for i in line.split(",")[1:]]
                data.append(listLine)
            dataNP = np.array(data)
            dataTP = dataNP.T
            dataTPL = dataTP.tolist()
            newdata = []
            for col in dataTPL:
                newdata.append(np.mean(col))
            print(str(cur), [(" %4f " % i) for i in newdata])

    plt.title(sort[0])
    plt.legend(loc=sort[2])
    plt.show()