from curriculum import Curriculum
import neuralobject

sampleSize = 6
additive = "lang_"

net = neuralobject.Net()
# curs = [(0,0,3),(0,0,4),(0,0,6),(0,0,7),(1,1,1),(1,2,4),(2,0,2),(2,2,2),(4,2,1),(6,0,1)]
curs = [(0,0,23)]
# curs = [(1,5,9), (9,4,2), (1,3,11)]
# curs=[(1,1,13)]
# curs=[(1,2,3),(1,1,4),(0,0,10),(2,2,2),()]

# curs = [(0,0,15),(1,2,12)]
for i in range(len(curs)):

    name = str(curs[i])
    cur = [("easy_train", curs[i][0], 0.05), ("middle_train", curs[i][1], 0.05), ("hard_train", curs[i][2], 0.05)]

    for s in range(sampleSize):

        try:

            print(name + " sample #" + str(s))

            net.reset()

            curriculum = Curriculum(cur, 4, "hard_complete_train", "results/"+additive + name, net,"hard_test", entry=s)

            curriculum.doCurricullumForScore()
        except:
            print(name + " sample #" + str(s))

            net = neuralobject.Net()

            curriculum = Curriculum(cur, 4, "hard_complete_train", "results/" + additive + name, net, "hard_test",
                                    entry=s)

            curriculum.doCurricullumForScore()





