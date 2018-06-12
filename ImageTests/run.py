from Thesis_BSc.ImageTests.curriculum import Curriculum
import Thesis_BSc.ImageTests.neuralobject as neuralobject
import traceback

sampleSize = 65
additive = "addaptive_"

net = neuralobject.Net()
curs = [(0.1,0.05,-20)]

for i in range(len(curs)):

    name = str(curs[i])
    cur = [("easy_train", curs[i][0], curs[i][0]), ("middle_train", curs[i][1], curs[i][1]), ("hard_train", curs[i][2], curs[i][2])]



    # name = str("addap_precise")

    for s in range(sampleSize):

        try:

            print(name + " sample #" + str(s))

            net.reset()

            curriculum = Curriculum(cur, 4, "test", "results/"+additive + name, net,"validation", entry=s)

            curriculum.doCurricullumAdaptive(23)
            # curriculum.doCurricullumForScore()
        except:
            print("crashed, restart")
            print(traceback.extract_stack())
            print(name + " sample #" + str(s))

            net = neuralobject.Net()

            curriculum = Curriculum(cur, 4, "test", "results/" + additive + name, net, "validation",
                                    entry=s)

            # curriculum.doCurricullumForScore()

            curriculum.doCurricullumAdaptive(23)


# cur = Curriculum((), 4, "hard_complete_train", "results/" + additive, net, "hard_test")
# print("Resulterend curriculum :", cur.createCurriculumAStar(83.5, 23*8, [0.8,0.4]))
