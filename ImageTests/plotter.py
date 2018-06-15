
import matplotlib.pyplot as plt
import numpy as np
import math

dingen = [
    "lang_(0, 0, 23)",
    "lang_(1, 2, 20)",
    "addaptive_(0.1, 0.05, -20)",



    # "auto_08052018"

]


##################

agentNames = ["No Curriculum", "Fixed Curriculum", "Addaptive Curriculum", "Unsupervised Curriculum"]

fig, ax = plt.subplots(1)
import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

for iii, sort in enumerate([("Accuracy over Epochs on Validationset", 100, "lower right", "validationset")]):#, ("Lossfunction over Epochs on Validationset", 750, "upper right", "loss")]):

    curs = dingen
    for iiii, cur in enumerate(curs):
        plt.title(str(cur))
        data = []
        for line in open("results/"+str(cur)+"_score_in_time_"+sort[3]+".csv"):
            listLine = [float(i) for i in line.split(",")[1:]]
            data.append(listLine)
        dataNP = np.array(data)
        dataTP = dataNP.T
        dataTPL = dataTP.tolist()
        newdata = []
        stdev = []

        betrouwbaarheids = 0

        for col in dataTPL:
            betrouwbaarheids = math.sqrt(len(col))
            stdev.append(np.std(col))
            newdata.append(np.mean(col))

        newdata2 = []
        stdev2 = []
        for yy in range(int(len(newdata)/5)):
            newdata2.append(np.mean(newdata[yy*5:(yy+1)*5]))
            stdev2.append(np.mean(stdev[yy*5:(yy+1)*5]))
        newdata = newdata2
        stdev = stdev2

        plt.axis((int(len(newdata)*0.0), len(newdata)-1, 00, sort[1]))

        # betrouwbaarheids *= 1.1

        # print betrouwbaarheids

        print (betrouwbaarheids)

        print (np.mean(stdev), np.mean(2*(np.array(stdev)/betrouwbaarheids)), "banana")

        a = np.array(newdata)-2*(np.array(stdev)/betrouwbaarheids)
        b = np.array(newdata)+2*(np.array(stdev)/betrouwbaarheids)

        ax.fill_between(range(len(newdata)), a, b, alpha=0.5)

        for ag in range(len(stdev)):
            if(not ag % len(dingen) == iiii):
                stdev[ag] = 0

        # plt.errorbar(range(len(newdata)), newdata, 2*(np.array(stdev)/betrouwbaarheids), label=agentNames[iiii],  alpha=0.5, capsize=2, marker = "o", markersize=4.0)

        plt.plot(range(len(newdata)), newdata, label=agentNames[iiii])

        if (iii == 0):
            data = []
            for line in open("results/" + str(cur) + "_score_in_end.csv"):
                listLine = [float(i) for i in line.split(",")[1:]]
                data.append(listLine)
            dataNP = np.array(data)
            dataTP = dataNP.T
            dataTPL = dataTP.tolist()
            newdata = []
            for col in dataTPL:
                newdata.append(np.mean(col))
            print(str(cur), [(" %4f " % i) for i in newdata])


    plt.xlabel('Epochs')
    plt.ylabel(sort[0].split(" ")[0])
    plt.title(sort[0])
    plt.legend(loc=sort[2])
    plt.show()
