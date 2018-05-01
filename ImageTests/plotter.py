import matplotlib.pyplot as plt
import numpy as np

dingen = [
    # (9, 18, 27),
    # (0,0,35),
    # (0,15,20),
    # (5,5,25),
    # (4,4,4)
# (5,0,20), (0,0,25), (0,5,20), (2,3,20), (1,2,22)
#     (1,2,42),(0,0,45),(10,10,25)
    (0, 0, 15), (1, 2, 12), (2,4,9)
    # ()
#     (0,0,15),
#     (0,0,18),
#     (5,5,5),
#     (3,6,9),
#     (9,6,3),
#     (9,0,9)
]
#
# for ding in dingen:
#     folder = "results/"
#     additive = "lang_"
#     file = str(ding)+"_score_in_time.csv"
#
#     plt.title((str(ding)))
#
#     f = open(folder+additive+file, "r")
#
#     lijn = [float(x) for x in f.readline().split(",")][1:]
#
#     plt.axis((0,170,0,750))
#
#     plt.plot(lijn)
#
#     plt.show()



##################

for iii, sort in enumerate([("validationset", 90, "lower right"), ("loss", 750, "upper right")]):

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

        plt.axis((int(len(newdata)*0.0), len(newdata), 60, sort[1]))

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