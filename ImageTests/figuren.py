

from random import randint, choice, uniform
import numpy as np


def decideOnChance(chance):
    return (((randint(0,100)/100.1)+0.01) < chance)



class Figuren():
    
    
    
    def __init__(self):
        print("Figuren Generator Actief")
        self.colorChoices = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.black = [0.9,0.9,0.9]
        self.white = [0.1,0.1,0.1]


    def checkPicture(self, picture, picturSize, percentage =0.035):
        limit = percentage*picturSize*3
        counter = 0
        for y in picture:
            for x in y:
                for col in x:
                    if (col == self.black[0]):
                        counter += 1
        return (counter >= limit)


    def fillFigure(self, picture, grain=0.0):
        pass

    def rotatePicture(self, picture):
        pass

    def distanceToLine(self, point, pointOne, pointTwo):
        return (np.linalg.norm(np.cross(pointTwo - pointOne, pointOne - point)) / np.linalg.norm(
            pointTwo - pointOne))

    def distance(self, pointOne, pointTwo):
        return np.linalg.norm(np.array(pointOne)-pointTwo)

    def elipse(self, pictureSize = 100, grain=0.0, elipse=0.0):

        if (decideOnChance(1.0-elipse)):
            return self.cirkelEmpty(pictureSize = pictureSize, grain=grain)

        picture = []

        startpos = (0, 0)
        maxSizeDiameter = 0
        minSizeDiameter=0
        edgeSize = randint(2, int(pictureSize / 7))
        while (maxSizeDiameter < 4 or minSizeDiameter < 4):
            startpos = (
                randint(int(pictureSize / 10), int((9 * pictureSize) / 10)),
                randint(int(pictureSize / 10), int((9 * pictureSize) / 10)))
            possibleDiameter = [startpos[0] - 1 - edgeSize, pictureSize - startpos[0] - 1 - edgeSize, startpos[1] - 1 - edgeSize,
                 pictureSize - startpos[1] - 1 - edgeSize]
            minSizeDiameter = np.min(possibleDiameter
                )
            maxSizeDiameter = np.max(possibleDiameter)

        diameter1 = randint(2, maxSizeDiameter)
        diameter2 = randint(2, minSizeDiameter)

        for y in range(pictureSize):
            rij = []
            for x in range(pictureSize):

                grainChance = decideOnChance(grain)

                if (grainChance):
                    rij.append(self.black)
                else:

                    # d1 = np.linalg.norm(startpos - np.array([x, y]))
                    # d2 = np.linalg.norm(startpos- np.array([x,y]))
                    d3 = np.linalg.norm(startpos- np.array([x,y]))


                    f1 = ((x-startpos[0])**2)/(diameter1**2)
                    f2 = ((y-startpos[1])**2)/(diameter2**2)
                    f3 =((x - startpos[0]- edgeSize) ** 2) / (diameter1 ** 2)
                    f4 = ((y - startpos[1]- edgeSize) ** 2) / (diameter2 ** 2)
                    f5 = ((x - startpos[0] + edgeSize) ** 2) / (diameter1 ** 2)
                    f6 = ((y - startpos[1] + edgeSize) ** 2) / (diameter2 ** 2)



                    # d1 = abs(startpos[0]-x)
                    # d2 = abs(startpos[1]-x)

                    if ( (((f3+f4) <= 1) or (f5+f6 <= 1) or ((f3+f6) <= 1) or ((f5+f4) <= 1))and not f1+f2 <= 1):
                        rij.append(self.white)
                    else:
                        if (decideOnChance(grain / (0.25 * (f1 + f2 + 0.1)))):
                            color = choice(self.colorChoices)
                            rij.append([color, color, color])
                        else:
                            rij.append(self.black)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def randomTriangle(self, pictureSize = 100, grain=0.0, randomTriangle=0.0):
        if (decideOnChance(1.0-randomTriangle)):
            return self.triangleEmpty(pictureSize = pictureSize, grain=grain)

        picture = []


        edgeSize = randint(2, int(pictureSize / 8))

        pointOne, pointTwo, pointThree = \
            np.array([randint(0, pictureSize),
                      randint(0, pictureSize)]), \
            np.array([randint(0, pictureSize),
                      randint(0, pictureSize)]), \
            np.array([randint(0, pictureSize),
                      randint(0, pictureSize)])
        diameter = 0
        while (diameter < 1):
            diameter = np.linalg.norm(choice([pointTwo,pointOne,pointThree])-choice([pointTwo,pointOne,pointThree]))

        reduceFactor = 0.85
        if (edgeSize < 3):
            reduceFactor = 0.75
        elif (edgeSize > 3):
            reduceFactor = 0.95

        if (diameter > 10):
            reduceFactor = reduceFactor / 1.1
        elif (diameter < 5):
            reduceFactor = reduceFactor * 1.1

        for y in range(pictureSize):
            rij = []
            for x in range(pictureSize):

                grainChance = decideOnChance(grain)

                if (grainChance):
                    rij.append(self.black)
                else:
                    p3 = np.array([x, y])

                    d1 = np.linalg.norm(np.cross(pointTwo - pointOne, pointOne - p3)) / np.linalg.norm(pointTwo - pointOne)
                    d2 = np.linalg.norm(np.cross(pointThree - pointOne, pointOne - p3)) / np.linalg.norm(
                        pointThree - pointOne)
                    d3 = np.linalg.norm(np.cross(pointThree - pointTwo, pointTwo - p3)) / np.linalg.norm(
                        pointThree - pointTwo)
                    if (
                            (d1 < edgeSize and d2 < diameter * reduceFactor and d3 < diameter * reduceFactor) or
                            (d2 < edgeSize and d1 < diameter * reduceFactor and d3 < diameter * reduceFactor) or
                            (d3 < edgeSize and d2 < diameter * reduceFactor and d1 < diameter * reduceFactor)):
                        rij.append(self.white)
                    else:
                        if (decideOnChance(grain / (0.25 * (d1+d2+d3 +0.1)))):
                            color = choice(self.colorChoices)
                            rij.append([color, color, color])
                        else:
                            rij.append(self.black)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def cirkelEmpty(self, pictureSize=100, grain=0.0):
        picture = []

        startpos = (0, 0)
        maxSizeDiameter = 0
        edgeSize = randint(2, int(pictureSize / 5))
        while (maxSizeDiameter < 4):
            startpos = (
            randint(int(pictureSize / 10), int((9 * pictureSize) / 10)), randint(int(pictureSize / 10), int((9 * pictureSize) / 10)))
            maxSizeDiameter = np.min(
                [startpos[0] - 1 - edgeSize, pictureSize - startpos[0] - 1 - edgeSize, startpos[1] - 1 - edgeSize,
                 pictureSize - startpos[1] - 1 - edgeSize])

        diameter = randint(2, maxSizeDiameter)

        for y in range(pictureSize):
            rij = []
            for x in range(pictureSize):
                grainChance = decideOnChance(grain)

                if (grainChance):
                    rij.append(self.black)
                else:
                    afstand = np.linalg.norm(startpos - np.array([x, y]))
                    if (afstand > diameter and afstand < diameter + edgeSize):
                        rij.append(self.white)
                    else:
                        if (decideOnChance(grain / (0.20 * (afstand +0.1)))):
                            color = choice(self.colorChoices)
                            rij.append([color, color, color])
                        else:
                            rij.append(self.black)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def squareEmpty(self, pictureSize=100, grain=0.0):
        picture = []

        startPosition = (0,0)
        maxSizeDiameter = 0
        edgeSize = randint(2, int(pictureSize / 5))
        while(maxSizeDiameter < 4):
            startPosition = (randint(int(pictureSize / 10), int((9 * pictureSize) / 10)), randint(int(pictureSize / 10), int((9 * pictureSize) / 10)))
            maxSizeDiameter = np.min([startPosition[0] - 1 - edgeSize, pictureSize - startPosition[0] - 1 - edgeSize, startPosition[1] - 1 - edgeSize, pictureSize - startPosition[1] - 1 - edgeSize])

        diameter = randint(2, maxSizeDiameter)

        for y in range(pictureSize):
            rij = []
            for x in range(pictureSize):

                grainChance = decideOnChance(grain)

                if (grainChance):
                    rij.append(self.black)
                else:
                    Xafstand = abs(x - startPosition[0])
                    Yafstand = abs(y - startPosition[1])
                    if (((Xafstand > diameter) or (Yafstand > diameter)) and (Xafstand < diameter + edgeSize) and (
                            Yafstand < diameter + edgeSize)):
                        rij.append(self.white)
                    else:
                        if (decideOnChance(grain / (0.15 * (Xafstand+Yafstand + 0.1)))):
                            color = choice(self.colorChoices)
                            rij.append([color, color, color])
                        else:
                            rij.append(self.black)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def triangleEmpty(self, pictureSize=100, grain=0.0):
        picture = []

        pointOne = (0, 0)
        maxSizeDiameter = 0
        edgeSize = randint(2, int(pictureSize / 8))
        while (maxSizeDiameter < 4):
            pointOne = (randint(int(pictureSize / 10), int((9 * pictureSize) / 10)),
                             randint(int(pictureSize / 10), int((9 * pictureSize) / 10)))
            maxSizeDiameter = np.max([pointOne[0] - 1 - edgeSize, pictureSize - pointOne[0] - 1 - edgeSize,
                                      pointOne[1] - 1 - edgeSize, pictureSize - pointOne[1] - 1 - edgeSize])

        diameter = randint(int(maxSizeDiameter/5), maxSizeDiameter)

        possiblePointsfromOne = set()

        for i in range(pictureSize):
            for j in range(pictureSize):
                afstand = np.linalg.norm(np.array([i,j])-pointOne)
                if (afstand*1.1 > diameter and afstand*0.90 < diameter ):
                    possiblePointsfromOne.add((i,j))

        pointTwo = choice(list(possiblePointsfromOne))

        possiblePointsfromTwo = set()


        for i in range(pictureSize):
            for j in range(pictureSize):
                afstand = np.linalg.norm(np.array([i,j])-pointTwo)
                if (afstand*1.1 > diameter and afstand*0.90 < diameter):
                    possiblePointsfromTwo.add((i,j))

        possiblePointsForThree = list(possiblePointsfromOne.intersection(possiblePointsfromTwo))


        pointThree = 0
        try:
            pointThree = choice(possiblePointsForThree)
        except:
            shortestLength = 1000*pictureSize
            for elem in possiblePointsfromOne:
                for elem2 in possiblePointsfromTwo:
                    distance = np.linalg.norm(np.array(list(elem))-np.array(list(elem2)))
                    if (distance < shortestLength):
                        pointThree = elem
                        shortestLength = distance
            



        pointOne = np.array(list(pointOne))
        pointTwo = np.array(list(pointTwo))
        pointThree = np.array(list(pointThree))


        # reduceFactor = 1.0-(1/(edgeSize))*(1.0-((maxSizeDiameter-diameter)/maxSizeDiameter))

        reduceFactor = 0.85
        if (edgeSize < 3):
            reduceFactor = 0.75
        elif (edgeSize > 3):
            reduceFactor = 0.95

        if (diameter > 10):
            reduceFactor = reduceFactor/1.1
        elif(diameter < 5):
            reduceFactor = reduceFactor*1.1

        for y in range(pictureSize):
            rij = []
            for x in range(pictureSize):

                grainChance = decideOnChance(grain)

                if (grainChance):
                    rij.append(self.black)
                else:
                    p3 = np.array([x, y])

                    # d1 = np.linalg.norm(np.cross(pointTwo - pointOne, pointOne - p3)) / np.linalg.norm(
                    #     pointTwo - pointOne)

                    d1 = self.distanceToLine(p3, pointOne,pointTwo)
                    d2 = self.distanceToLine(p3, pointOne, pointThree)
                    d3 = self.distanceToLine(p3, pointThree, pointTwo)
                    # d2 = np.linalg.norm(np.cross(pointThree - pointOne, pointOne - p3)) / np.linalg.norm(
                    #     pointThree - pointOne)
                    # d3 = np.linalg.norm(np.cross(pointThree - pointTwo, pointTwo - p3)) / np.linalg.norm(
                    #     pointThree - pointTwo)
                    if (
                            (d1 < edgeSize and d2 < diameter * reduceFactor and d3 < diameter * reduceFactor) or
                            (d2 < edgeSize and d1 < diameter * reduceFactor and d3 < diameter * reduceFactor) or
                            (d3 < edgeSize and d2 < diameter * reduceFactor and d1 < diameter * reduceFactor)):
                        rij.append(self.white)
                    else:
                        if (decideOnChance(grain / (0.25 * (d1 + d2 + d3 + 0.1)))):
                            color = choice(self.colorChoices)
                            rij.append([color, color, color])
                        else:
                            rij.append(self.black)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def filled(self, pictureSize=100):

        picture = []


        for x in range(pictureSize):
            rij = []
            for y in range(pictureSize):
                rij.append(self.white)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def empty(self, pictureSize=100):
        picture = []
        for x in range(pictureSize):
            rij = []
            for y in range(pictureSize):
                rij.append(self.black)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def parralellogram(self, pictureSize=100, parralellogram=0.0, grain=0.0):
        if (decideOnChance(1.0-parralellogram)):
            return self.squareEmpty(pictureSize = pictureSize, grain=grain)

        picture = []

        pointOne = (0, 0)
        maxSizeDiameter = 0
        edgeSize = randint(4, int(pictureSize / 6))
        while (maxSizeDiameter < 6):
            pointOne = (
                randint(int(pictureSize / 10), int((9 * pictureSize) / 10)),
                randint(int(pictureSize / 10), int((9 * pictureSize) / 10)))
            possibleDiameter = [pointOne[0] - 1 - edgeSize, pictureSize - pointOne[0] - 1 - edgeSize, pointOne[1] - 1 - edgeSize,
                 pictureSize - pointOne[1] - 1 - edgeSize]
            maxSizeDiameter = np.min(possibleDiameter)

        diameter = randint(4, maxSizeDiameter)
        pointTwo = []
        for y in range(pictureSize):
            for x in range(pictureSize):
                d = self.distance(np.array([x,y]),pointOne)
                if (d > diameter and d < diameter+edgeSize):
                    pointTwo.append([x,y])

        pointTwo = np.array(choice(pointTwo))
        pointThree = np.array([
            randint(int(pictureSize / 10), int((9 * pictureSize) / 10)),
            randint(int(pictureSize / 10), int((9 * pictureSize) / 10))
        ])
        pointFour = pointTwo+pointThree-pointOne

        l1 = (np.linalg.norm(pointOne-pointTwo))
        l2 = (np.linalg.norm(pointOne-pointThree))
        l3 = (np.linalg.norm(pointThree - pointFour))
        l4 = (np.linalg.norm(pointFour-pointTwo))

        trueSize = l1*(self.distanceToLine(pointThree, pointOne, pointTwo))
        trueEdgeSize = (edgeSize+l1)*(self.distanceToLine(pointThree, pointOne, pointTwo)+edgeSize)

        for y in range(pictureSize):
            rij = []
            for x in range(pictureSize):

                grainChance = decideOnChance(grain)

                if (grainChance):
                    rij.append(self.black)
                else:

                    p = np.array([x,y])

                    dr1 = (np.linalg.norm(np.cross(pointTwo - pointOne, pointOne - p)) / np.linalg.norm(
                        pointTwo - pointOne))
                    dr2 = (np.linalg.norm(np.cross(pointThree - pointOne, pointOne - p)) / np.linalg.norm(
                        pointThree - pointOne))
                    dr3 = (np.linalg.norm(np.cross(pointFour - pointThree, pointThree - p)) / np.linalg.norm(
                        pointFour - pointThree))
                    dr4 = (np.linalg.norm(np.cross(pointTwo - pointFour, pointFour - p)) / np.linalg.norm(
                        pointTwo - pointFour))


                    triangle1 = 0.5*dr1*l1
                    triangle2 = 0.5*dr2*l2
                    triangle3 = 0.5*dr3*l3
                    triangle4 = 0.5*dr4*l4

                    triangle5 = 0.5 * (dr1 + edgeSize) * l1
                    triangle6 = 0.5 * (dr2 + edgeSize) * l2
                    triangle7 = 0.5 * (dr3 + edgeSize) * l3
                    triangle8 = 0.5 * (dr4 + edgeSize) * l4


                    pointSize = (triangle1+triangle2+triangle3+triangle4)
                    withEdgeSize = (triangle5+triangle6+triangle7+triangle8)


                    if (not pointSize <= trueSize
                        and
                        withEdgeSize <= trueEdgeSize
                            ):
                        rij.append(self.white)
                    else:
                        if (decideOnChance(grain / (0.25 * (pointSize + 0.1)))):
                            color = choice(self.colorChoices)
                            rij.append([color, color, color])
                        else:
                            rij.append(self.black)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def line(self, pictureSize=100, bended=0.0, grain=0.0):

        if (decideOnChance(bended)):
            return self.sinusoide(pictureSize=pictureSize, grain = grain)

        picture = []

        pos1 = [randint(0, pictureSize), randint(0, pictureSize)]
        pos2 = [randint(0, pictureSize), randint(0, pictureSize)]

        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        edgeSize = randint(1, int(pictureSize / 8))

        for y in range(pictureSize):
            rij = []
            for x in range(pictureSize):

                grainChance = decideOnChance(grain)

                if (grainChance):
                    rij.append(self.black)
                else:

                    p3 = np.array([x,y])
                    d1 = self.distanceToLine(p3, pos1, pos2)
                    if (d1 < edgeSize ):
                        rij.append(self.white)
                    else:
                        
                        if(decideOnChance(grain/(0.5*(d1+1)))):
                            color = choice([0.1,0.2,0.3,0.4,0.5])
                            rij.append([color, color, color])
                        else:
                            rij.append(self.black)
            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))

    def sinusoide(self, pictureSize=100, grain=0.0):

        picture = []

        pos1 = [randint(0, pictureSize), randint(0, pictureSize)]
        pos2 = [randint(0, pictureSize), randint(0, pictureSize)]

        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        edgeSize = randint(2, int(pictureSize / 8))

        for y in range(pictureSize):
            rij = []
            for x in range(pictureSize):

                grainChance = decideOnChance(grain)

                if (grainChance):
                    rij.append(self.black)
                else:

                    a = (pos1[1]-pos2[1])/(pos1[0]-pos2[0])
                    b = pos1[1]-a*pos1[0]

                    sinWaarde = a*x+b+edgeSize*np.sin(((2*np.pi)/pictureSize)*x)

                    if ((y > sinWaarde- edgeSize) and (y < sinWaarde+edgeSize)):
                        rij.append(self.white)

                    else:
                        if(decideOnChance(grain/(4))):
                            color = choice(self.colorChoices)
                            rij.append([color, color, color])
                        else:
                            rij.append(self.black)

            picture.append(rij)
        return np.array(picture).transpose((2, 0, 1))