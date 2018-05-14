import matplotlib.pyplot as plt
import numpy as np

dingen = [
    "addaptive_(0.1, 0.05, -0.01)"
    ,
    "adda_(0.03, 0.03, -2)",
"adda_(0.0, 0.0, -2)",
"adda_(0.01, 0.01, -2)",
# "adda_(0.1, 0.05, -2)",
"adda_(0.01, 0.1, -2)",

    # "lang_"+str((0,0,23))
    # ,"lang_"+str((1,2,20))
    # ,"auto_08052018"

]


##################

agentNames = ["No Curriculum", "Supervised Curriculum", "Addaptive Curriculum", "Unsupervised Curriculum"]

for iii, sort in enumerate([("validationset", 86, "lower right"), ("loss", 750, "upper right")]):

    curs = dingen
    for iiii, cur in enumerate(curs):
        plt.title(str(cur))
        data = []
        for line in open("results/"+str(cur)+"_score_in_time_"+sort[0]+".csv"):
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

        plt.axis((int(len(newdata)*0.0), len(newdata)-1, 70, sort[1]))

        plt.plot(newdata, label=str(cur))


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

    plt.title(sort[0])
    plt.legend(loc=sort[2])
    plt.show()