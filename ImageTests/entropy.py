


from random import randint, choice
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy


class Entropy(object):



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

        positions.append(startpos)

        a = positions
        positions = np.array((a - np.max(a))/-np.ptp(a))

        # plt.plot(positions[:,0], positions[:,1])
        # plt.show()

        return positions

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

        positions.append(startpos)

        a = positions
        positions = np.array((a - np.max(a))/-np.ptp(a))

        # plt.plot(positions[:,0], positions[:,1])
        # plt.show()

        return positions

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

        positions.append(startpos)

        a = positions
        positions = np.array((a - np.max(a))/-np.ptp(a))

        # plt.plot(positions[:,0], positions[:,1])
        # plt.show()

        return positions

    def __init__(self):
        self.locations = {}
        for i, func in enumerate([self.koch, self.koch2, self.koch3]):
            self.locations[i] = []
            for x in range(6):
                self.locations[i].append(func(x))


e = Entropy()



class tsp_instance(object):
    def __init__(self, order, entropydegree=0.0):
        self.order = order

        self.entropy = entropydegree
        self.locations = []
        pickKoch = 0#randint(0,len(e.locations)-1)
        kochline = list(deepcopy(e.locations[pickKoch]))
        i = order
        while(i >= len(kochline[0])):
            i-= len(kochline[0])
            line = kochline.pop(0)
            for elem in line:
                self.locations.append(elem)

        for j in range(i):
            elem = list(kochline[0]).pop(randint(0,len(kochline[0])-1))
            self.locations.append(elem)


        self.locations = np.array(self.locations)

        self.shake()


    def shake(self):
        print (self.locations)
        plt.plot(self.locations[:, 0], self.locations[:, 1])
        plt.show()

    def getTensor(self):
        pass


for x in range(2,6):
    tsp_instance(x)