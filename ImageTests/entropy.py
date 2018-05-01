


import turtle

import numpy as np
from matplotlib import pyplot as plt



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

        koch_flake = "FRFRFRFRF"

        for i in range(order):
            # if (i%2 == 0):
            koch_flake = koch_flake.replace("F", "FLFRFRFLF")
            # koch_flake = koch_flake.replace("F", "FLFFLLFLLFFLF")
            #
            # else:
            # #     # koch_flake = koch_flake.replace("F", "FRFLFLFRF")
            #     koch_flake = koch_flake.replace("F", "FRFLFRF")


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

        plt.plot(positions[:,0], positions[:,1])
        plt.show()

        return positions




    def __init__(self):
        pass


    #
    # def locationsOnKoch(self, limit, entropy):
    #     pass
    #
    # def locationsOnCirkle(self, limit, entropy):
    #     pass
    #
    # def locationsOnStar(self, limit, entropy):
    #     pass





for x in range(6):
    print (len(Entropy().koch(x)))


