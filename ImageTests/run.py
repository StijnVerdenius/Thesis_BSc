from curriculum import Curriculum
import neuralobject
import traceback

sampleSize = 10
additive = "adda_"

net = neuralobject.Net()
# curs = [(0,0,3),(0,0,4),(0,0,6),(0,0,7),(1,1,1),(1,2,4),(2,0,2),(2,2,2),(4,2,1),(6,0,1)]
curs = [(0.0,0.0,-2)]
# curs = [(1,5,9), (9,4,2), (1,3,11)]
# curs=[(1,1,13)]
# curs=[(1,2,3),(1,1,4),(0,0,10),(2,2,2),()]

# automated = [[0.0, 0.0], [0.1, 0.05], [0.3, 0.05], [0.3, 0.05], [0.5, 0.1], [0.7, 0.2], [0.7, 0.2], [0.7, 0.2], [0.7, 0.15], [0.6, 0.1], [0.7, 0.2], [0.7, 0.3], [0.7, 0.25], [0.6, 0.2], [0.6, 0.2], [0.7, 0.3], [0.7, 0.2], [0.7, 0.3], [0.6, 0.35], [0.7, 0.3], [0.7, 0.35], [0.7, 0.35], [0.7, 0.35], [0.7, 0.35]]

# addaptive = [[0.0, 0.0], [0.1, 0.05], [0.3, 0.05], [0.3, 0.15], [0.4, 0.2], [0.5, 0.25], [0.6, 0.3], [0.7, 0.3]]


# cur = [tuple([str(tuple(x))+"_train", 1, 0.05]) for x in addaptive] + [tuple([str((0.75, 0.32))+"_train", 20, -1.0])]

# curs = [(0,0,15),(1,2,12)]
for i in range(len(curs)):

    name = str(curs[i])
    cur = [("easy_train", curs[i][0], curs[i][0]), ("middle_train", curs[i][1], curs[i][1]), ("hard_train", curs[i][2], curs[i][2])]



    # name = str("addap_precise")

    for s in range(sampleSize):

        try:

            print(name + " sample #" + str(s))

            net.reset()

            curriculum = Curriculum(cur, 4, "hard_complete_train", "results/"+additive + name, net,"hard_test", entry=s)

            curriculum.doCurricullumAdaptive(23)
            # curriculum.doCurricullumForScore()
        except:
            print("crashed, restart")
            print(traceback.extract_stack())
            print(name + " sample #" + str(s))

            net = neuralobject.Net()

            curriculum = Curriculum(cur, 4, "hard_complete_train", "results/" + additive + name, net, "hard_test",
                                    entry=s)

            # curriculum.doCurricullumForScore()

            curriculum.doCurricullumAdaptive(23)


# cur = Curriculum((), 4, "hard_complete_train", "results/" + additive, net, "hard_test")
# print("Resulterend curriculum :", cur.createCurriculumAStar(83.5, 23*8, [0.8,0.4]))
