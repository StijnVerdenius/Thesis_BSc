


from random import randint, choice
import numpy as np
# from matplotlib import pyplot as #plt
from copy import deepcopy
from sklearn import preprocessing
import torch

mima = preprocessing.MinMaxScaler()



class Entropy(object):


    def normalize(self, a):
        return mima.fit_transform(a)



    def rotationmatrix(self, degrees):
        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    def koch(self, order):

        sideLength = 100/(3**order)

        startpos = np.array([0,0])

        pointer = np.array([0, 0])
        richting = np.array([1, 0])

        positions = [startpos]

        koch_flake = "FRFRF"

        for i in range(order):
            koch_flake = koch_flake.replace("F", "FLFRFLF")

        for i, move in enumerate(koch_flake):
            if move == "F":
                pointer = pointer + richting * sideLength
            elif move == "L":
                richting = np.dot(self.rotationmatrix(60),richting)
                positions.append(pointer)
            elif move == "R":
                richting = np.dot(self.rotationmatrix(-120), richting)
                positions.append(pointer)

        

        positions=  self.normalize(positions)

        if (self.show):
            print(len(positions))
            # positions = noisify(np.concatenate((positions, np.array([positions[0]])), axis=0), 1.0)
            #plt.scatter(positions[:,0], positions[:,1])
            #plt.show()
            #plt.plot(positions[:,0], positions[:,1])
            #plt.show()
        else:
            return [list([float(y) for y in x]) for x in positions]


    def koch2(self, order):

        sideLength = 100/(3**order)

        startpos = np.array([0,0])

        pointer = np.array([0, 0])
        richting = np.array([1, 0])

        positions = [startpos]

        koch_flake = "FRFRF"

        for i in range(order):
            koch_flake = koch_flake.replace("F", "FLFFLLFLLFFLF")

        for i, move in enumerate(koch_flake):
            if move == "F":
                pointer = pointer + richting * sideLength
            elif move == "L":
                richting = np.dot(self.rotationmatrix(60),richting)
                positions.append(pointer)
            elif move == "R":
                richting = np.dot(self.rotationmatrix(-120), richting)
                positions.append(pointer)

        
        

        positions=  self.normalize(positions)

        if (self.show):
            print (len(positions))
            positions = np.concatenate((positions, np.array([positions[0]])), axis=0)
            #plt.plot(positions[:,0], positions[:,1])
            #plt.show()
        else:
            return [list([float(y) for y in x]) for x in positions]


    def koch3(self, order):

        sideLength = 100/(3**order)

        startpos = np.array([0,0])

        pointer = np.array([0, 0])
        richting = np.array([1, 0])

        positions = [startpos]

        koch_flake = "FRFRFRF"

        for i in range(order):
            if(i%2 ==1):
                koch_flake = koch_flake.replace("F", "FLFRFRFLF")
            else:
                koch_flake = koch_flake.replace("F", "FRFLFLFRF")

        for i, move in enumerate(koch_flake):
            if move == "F":
                pointer = pointer + richting * sideLength
            elif move == "L":
                richting = np.dot(self.rotationmatrix(90),richting)
                positions.append(pointer)
            elif move == "R":
                richting = np.dot(self.rotationmatrix(-90), richting)
                positions.append(pointer)

        positions=  self.normalize(positions)

        if (self.show):
            print(len(positions))
            positions = np.concatenate((positions, np.array([positions[0]])), axis=0)
            #plt.plot(positions[:,0], positions[:,1])
            #plt.show()
        else:
            return [list([float(y) for y in x]) for x in positions]

    def __init__(self, show=False):
        self.show = show
        self.locations = {}
        for i, func in enumerate([self.koch, self.koch2, self.koch3]):
            self.locations[i] = []
            for x in range(5):

                self.locations[i].append(func(x))

                


e = Entropy(show=False)

class tsp_instance(object):
    def __init__(self, order, entropydegree=0.0):
        self.order = order
        self.entropy = entropydegree
        self.locations = []
        pickKoch = 0#randint(0,len(e.locations)-1)
        # kochline = deepcopy(e.locations[pickKoch])
        kochline = [[a for a in b] for b in e.locations[pickKoch]]
        # print kochline
        toPick = order
        bestLine = 0
        while (order>len(kochline[bestLine])):
            bestLine += 1

        pickedLine = kochline[bestLine]

        while (toPick > 0):
            elem = pickedLine.pop(randint(0,len(pickedLine)-1))
            self.locations.append(elem)
            toPick -= 1

        self.locations = np.array(self.locations)

        self.shake()

    def normalize(self, a):
        return mima.fit_transform(a)

    def noisify(self, pure, amount):
        noise = np.random.normal(0, 1, pure.shape)*(amount*0.15)
        temp = pure+noise
        temp2 = np.dot(temp, self.rotationmatrix(randint(0,359)))
        return self.normalize(temp2)

    def shake(self):
        self.locations = self.noisify(self.locations, self.entropy)
        # print (self.locations)
        # #plt.plot(self.locations[:, 0], self.locations[:, 1])
        # #plt.show()

    def getTensor(self):
        return torch.FloatTensor(self.locations)

class tsp_batch(object):
    def __init__(self, order, entropydegree, size):
        self.data = []
        for x in range(size):
            self.data.append(tsp_instance(order=order, entropydegree=entropydegree).getTensor())

    def getall(self):
        return self.data

    def __len__(self):
        return len(self.data)




# import cProfile, pstats, StringIO
# pr = cProfile.Profile()
# pr.enable()
# for aan in range(1000):
#     # print aan
#     tsp_instance(20)
# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()

# data = tsp_batch(20, 0, 5).getall()


# print(data)
# print(data[0].size(), len(data))

